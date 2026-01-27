from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class VectorManager:
    def __init__(self):
        """
        Disk dizini parametresini kaldırdık. 
        Veritabanı artık sadece RAM üzerinde yaşayacak.
        """
        # Yerel embedding modeli
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def create_vector_store(self, chunks):
        """Metin parçalarını metadata ile birlikte RAM üzerinde vektörleştirir."""
        print("Vektörleştirme işlemi RAM üzerinde başlatılıyor...")
        
        # 'persist_directory' parametresini sildik. 
        # Bu sayede Chroma verileri diske yazmaz, sadece bellekte tutar.
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        print("Vektör veritabanı başarıyla RAM'e yüklendi (Disk kullanılmadı).")
        return vector_db

    def get_vector_store(self):
        """
        Bellek içi yapıda bu metod genellikle doğrudan 'create_vector_store' 
        tarafından dönen nesne üzerinden kullanılır. 
        Ancak yapısal uyum için boş bir Chroma nesnesi döndürebilir veya 
        hiç kullanılmayabilir.
        """
        # Bellek içi yapıda veritabanını 'yüklemek' diye bir kavram yoktur, 
        # çünkü her seferinde sıfırdan oluşturulur.
        return None