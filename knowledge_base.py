import os
import time
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "catalogo_productos.csv")
FAISS_INDEX_DIR = os.path.join(os.path.dirname(__file__), "faiss_index")

EMBEDDING_MODEL = "models/gemini-embedding-001"


def load_pdfs(docs_dir: str) -> list:
    documents = []
    for filename in os.listdir(docs_dir):
        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(docs_dir, filename))
            documents.extend(loader.load())
    return documents


def load_csv_as_documents(csv_path: str) -> list:
    df = pd.read_csv(csv_path)
    documents = []
    for _, row in df.iterrows():
        content = (
            f"Producto: {row['nombre']}\n"
            f"Categoría: {row['categoria']}\n"
            f"Precio: ${row['precio']}\n"
            f"Descripción: {row['descripcion']}\n"
            f"Stock: {row['stock']} unidades"
        )
        from langchain_core.documents import Document
        documents.append(Document(
            page_content=content,
            metadata={"source": "catalogo_productos", "producto": row["nombre"]}
        ))
    return documents


def split_documents(documents: list, chunk_size=500, chunk_overlap=100) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(documents)


def build_vector_store(chunks: list) -> FAISS:
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    batch_size = 50
    vector_store = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"Procesando lote {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1} ({len(batch)} chunks)...")
        for attempt in range(3):
            try:
                if vector_store is None:
                    vector_store = FAISS.from_documents(batch, embeddings)
                else:
                    vs_batch = FAISS.from_documents(batch, embeddings)
                    vector_store.merge_from(vs_batch)
                break
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    wait = 45 * (attempt + 1)
                    print(f"  Cuota agotada. Esperando {wait}s...")
                    time.sleep(wait)
                else:
                    raise
    vector_store.save_local(FAISS_INDEX_DIR)
    return vector_store


def load_vector_store() -> FAISS:
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    if os.path.exists(FAISS_INDEX_DIR):
        return FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

    print("No se encontró índice FAISS. Construyendo desde los documentos...")
    pdf_docs = load_pdfs(DOCS_DIR)
    csv_docs = load_csv_as_documents(CSV_PATH)
    all_docs = pdf_docs + csv_docs
    chunks = split_documents(all_docs)
    print(f"Documentos cargados: {len(pdf_docs)} páginas PDF + {len(csv_docs)} productos CSV")
    print(f"Total de chunks generados: {len(chunks)}")
    return build_vector_store(chunks)


def get_retriever(vector_store: FAISS, k: int = 5):
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})


if __name__ == "__main__":
    vs = load_vector_store()
    print(f"\nVector store cargado con {vs.index.ntotal} vectores.")
