from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class VectorManager:
    def __init__(self, db_directory="./db_academic"):
        # Ücretsiz ve güçlü bir yerel embedding modeli
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db_directory = db_directory

    def create_vector_store(self, chunks):
        """Metin parçalarını metadata (sayfa no vb.) ile birlikte vektöre dönüştürür."""
        print("Vektörleştirme işlemi metadata ile birlikte başlıyor...")
        
        # 'from_texts' yerine 'from_documents' kullanılarak metadata korunur.
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_directory
        )
        
        print(f"Vektör veritabanı metadata ile birlikte '{self.db_directory}' dizinine kaydedildi.")
        return vector_db

    def get_vector_store(self):
        """Kaydedilmiş veritabanını yükler."""
        return Chroma(
            persist_directory=self.db_directory,
            embedding_function=self.embeddings
        )