from langchain_community.document_loaders import PyPDFLoader

def load_pdf_document(pdf_path):
    loader = PyPDFLoader(str(pdf_path))
    return loader.load()