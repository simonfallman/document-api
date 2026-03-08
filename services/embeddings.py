from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings


def get_embeddings() -> BedrockEmbeddings:
    return BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        region_name=settings.aws_region,
    )


def build_vectorstore(docs, collection_name: str, document_name: str = "") -> int:
    """
    Chunk, embed, and persist documents into ChromaDB.
    collection_name is the MD5 hash of the file — same file always maps to same collection.
    Returns the number of chunks created.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["document_name"] = document_name
        chunk.metadata["collection_hash"] = collection_name
        chunk.metadata["chunk_index"] = i

    vectorstore = Chroma(
        persist_directory=settings.chroma_dir,
        embedding_function=get_embeddings(),
        collection_name=collection_name,
    )
    vectorstore.add_documents(chunks)
    return len(chunks)


def get_vectorstore(collection_name: str) -> Chroma:
    return Chroma(
        persist_directory=settings.chroma_dir,
        embedding_function=get_embeddings(),
        collection_name=collection_name,
    )


def get_vectorstores(collection_names: list[str]) -> list[Chroma]:
    return [get_vectorstore(name) for name in collection_names]
