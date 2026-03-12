from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

# =====================
# Load .env to get API key
# =====================

load_dotenv()

# =============================
# Load Vector Database
# =============================

print("LOG: EMBEDDING_MODEL")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("LOG: LOAD_FAISS_INDEX")
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})

# =============================
# Prompt Template
# =============================

prompt_template = """
You are a helpful assistant specialized ONLY in Graph Theory.

Use the following context to answer the question.

Rules:
- Only answer questions related to Graph Theory.
- If the question is outside Graph Theory, say:
  "I can only answer questions related to Graph Theory."
- If the context does not contain the answer, say:
  "I don't know based on the provided Graph Theory context."

Context:
{context}

Question:
{question}

Answer:
"""

llm_prompt = """
You are a helpful assistant specialized in Graph Theory.

Question:
{question}

Answer:
"""

prompt_custom = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
) 

# =============================
# LLM
# =============================
print("LOG: LLM")
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0
)

# =============================
# Retrieval QA Chain
# =============================
print("LOG: RETRIEVAL_QA_CHAIN")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_custom},
    return_source_documents=True
)

# =============================
# FASTAPI
# =============================
print("LOG: FASTAPI")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask_question(data: Question):
    question = data.question

    answer = qa_chain.invoke({"query": question})

    return {
        "answer": answer["result"]
    }

@app.post("/ask-llm")
def ask_llm(data: Question):
    question = data.question

    answer = llm.invoke([
        HumanMessage(content=question)
    ]).content

    return {
        "answer": answer
    }