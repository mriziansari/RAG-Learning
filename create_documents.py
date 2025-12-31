from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentCreator:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_pdf(self):
        loader = PyPDFLoader(self.file_path)
        return loader.load()


    