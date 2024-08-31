from langchain_community.document_loaders import WebBaseLoader

url = "https://brainlox.com/courses/category/technical"
loader = WebBaseLoader(url)
documents = loader.load()

# Print the first document for inspection
print(documents[0].page_content)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Use a pre-trained sentence-transformer model for embeddings
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Generate embeddings
doc_embeddings = embedder.embed_documents([doc.page_content for doc in docs])

# Store embeddings in FAISS vector store
vector_store = FAISS.from_embeddings(embeddings=doc_embeddings, documents=docs)

# Save the vector store to disk
vector_store.save("vector_store.faiss")
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from langchain.chains import RetrievalQAChain
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

app = Flask(__name__)
api = Api(app)

# Load the vector store and embeddings
vector_store = FAISS.load_local("vector_store.faiss", HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
qa_chain = RetrievalQAChain.from_chain_type(llm="gpt-3", chain_type="qa", retriever=vector_store.as_retriever())

class ChatbotAPI(Resource):
    def post(self):
        user_query = request.json.get("query")
        if not user_query:
            return jsonify({"error": "No query provided"}), 400

        # Get response from the chatbot
        response = qa_chain.run(user_query)
        return jsonify({"response": response})

api.add_resource(ChatbotAPI, "/chatbot")

if __name__ == "__main__":
    app.run(debug=True)
