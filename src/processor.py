from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        # Profesyonel ayarlar: 1000 karakterlik bloklar, 200 karakterlik çakışma
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    def split_docs(self, documents):
        return self.splitter.split_documents(documents)