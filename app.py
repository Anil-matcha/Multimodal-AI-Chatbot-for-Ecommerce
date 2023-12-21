from flask import Flask, jsonify
from flask import request
from flask_cors import CORS
import chromadb
import os
import pandas as pd
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import ChromaVectorStore
from llama_index.retrievers import VectorIndexRetriever
from llama_index.retrievers import BM25Retriever
from llama_index.retrievers import BaseRetriever
from llama_index.embeddings import GeminiEmbedding
from llama_index.llms import Gemini
from llama_index.prompts import PromptTemplate
from llama_index.llms import ChatMessage, Gemini
from llama_index.postprocessor import SentenceTransformerRerank
from llama_index.schema import TextNode
from llama_index import QueryBundle

from trulens_eval import Feedback, Tru, TruLlama, Huggingface
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.litellm import LiteLLM

class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query, **kwargs):
        bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)
        vector_nodes = self.vector_retriever.retrieve(query, **kwargs)

        # combine the two lists of nodes
        all_nodes = []
        node_ids = set()
        for n in bm25_nodes + vector_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        return all_nodes

qa_prompt = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "You are an AI Shopify help assistant, suggest me the best matching products for the query based on the context information and not prior knowledge. Be descriptive of why you suggested the options and mention attributes of each. Don't mention about context information in the response, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)

os.environ["GOOGLE_API_KEY"] = "api-key"

app = Flask(__name__)
CORS(app)

df = pd.read_csv("../image_descriptions.csv")
tnodes = []
for index, row in df.iloc[1:].iterrows():
    img, desc = row[0], row[1]
    text_node = TextNode()
    metadata = {}
    text_node.text = desc
    metadata["image"] = img
    text_node.metadata = metadata
    tnodes.append(text_node)

@app.route('/get_data')
def home():
    query = request.args.get("query")
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    embed_model = GeminiEmbedding(
        model_name="models/embedding-001", api_key=os.environ["GOOGLE_API_KEY"]
    )
    service_context = ServiceContext.from_defaults(llm=Gemini(api_key=os.environ["GOOGLE_API_KEY"]), embed_model=embed_model)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        service_context=service_context,
    )
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10,
    )
    bm25_retriever = BM25Retriever.from_defaults(nodes=tnodes, similarity_top_k=10)
    hybrid_retriever = HybridRetriever(retriever, bm25_retriever)
    reranker = SentenceTransformerRerank(top_n=10, model="BAAI/bge-reranker-base")


    nodes = hybrid_retriever.retrieve(query)
    reranked_nodes = reranker.postprocess_nodes(
    nodes,
    query_bundle=QueryBundle(
        query
    ),
    )
    context_str = "\n\n".join([n.text for n in reranked_nodes])
    content = qa_prompt.format(context_str=context_str, query_str=query)
    messages = [
        ChatMessage(role="user", content=content),
    ]
    response = Gemini().chat(messages).message.content
    print("Resp", type(response))
    images = [x.metadata["image"] for x in reranked_nodes]
    return jsonify(images=images, text=response)

if __name__ == '__main__':
    app.run(debug=True)
