from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents, chunk_size=1200, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    return text_splitter.split_documents(documents)
