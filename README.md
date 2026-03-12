# project-rag-fastapi-be
[Click here for Front-end](https://github.com/FeikVizeege/project-rag-react-fe)

## Setup
**Python**: 3.10

**Poetry**: Optional

Poetry: [Basic usage](https://python-poetry.org/docs/basic-usage/)

If we use Python to run directly, below is a list of dependencies:
- langchain (>=1.2.10,<2.0.0)
- langchain-community (>=0.4.1,<0.5.0)
- langchain-openai (>=1.1.10,<2.0.0)
- sentence-transformers (>=5.2.3,<6.0.0)
- python-dotenv (>=1.2.2,<2.0.0)
- faiss-cpu (>=1.13.2,<2.0.0)
- langchain-text-splitters (>=1.1.1,<2.0.0)
- langchain-classic (>=1.0.1,<2.0.0)
- langchain-groq (>=1.1.2,<2.0.0)
- fastapi (>=0.135.1,<0.136.0)
- uvicorn (>=0.41.0,<0.42.0)

## Run
Please run build_vector_db.py directly to build vector database before run API<br>
Python: `py run build_vector_db.py`<br>
Poetry: `poetry run build_vector_db.py`
