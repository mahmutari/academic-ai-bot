from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class VectorManager:
    def __init__(self, db_directory="./db_academic"):
        # Ücretsiz ve güçlü bir yerel embedding modeli seçiyoruz
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db_directory = db_directory

    def create_vector_store(self, chunks):
        """Metin parçalarını vektöre dönüştürür ve diske kaydeder."""
        print("Vektörleştirme işlemi başlıyor (Bu biraz sürebilir)...")
        vector_db = Chroma.from_texts(
            texts=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_directory
        )
        print(f"Vektör veritabanı '{self.db_directory}' dizinine kaydedildi.")
        return vector_db

    def get_vector_store(self):
        """Kaydedilmiş veritabanını yükler."""
        return Chroma(
            persist_directory=self.db_directory,
            embedding_function=self.embeddings
        )