from langchain_community.document_loaders import PyPDFLoader

class PDFLoader:
    def get_pdf_documents(self, file_path):
        """PDF'i sayfa sayfa okur ve metadata (sayfa no) ile birlikte döner."""
        loader = PyPDFLoader(file_path)
        # load() metodu bize her sayfası metadata içeren bir Document listesi verir
        documents = loader.load() 
        return documents