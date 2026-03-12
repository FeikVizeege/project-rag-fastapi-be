from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# =============================
# Load Dataset
# =============================
print("LOG: LOAD_DATASET")
loader = TextLoader("graph_dataset.txt")
documents = loader.load()

# =============================
# Split Documents
# =============================
print("LOG: SPLIT_DOCUMENTS")
text_splitter = CharacterTextSplitter(
    separator='---',
    chunk_size=512,
    chunk_overlap=100
)

docs = text_splitter.split_documents(documents)

# =============================
# Embedding Model
# =============================

print("LOG: EMBEDDING_MODEL")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =============================
# Vector Database (FAISS)
# =============================
print("LOG: FAISS")
vectorstore = FAISS.from_documents(docs, embeddings)

print("LOG: SAVE_FAISS_INDEX")
vectorstore.save_local("faiss_index")

print("DONE: FAISS index created")