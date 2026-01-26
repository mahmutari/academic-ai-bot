from src.loader import PDFLoader
from src.processor import TextProcessor
from src.vector_store import VectorManager

def main():
    pdf_path = "data/Python-plot.pdf"
    
    # 1. Yükle (Day 1)
    loader = PDFLoader()
    raw_text = loader.get_pdf_text(pdf_path)

    # 2. Parçala (Day 2)
    processor = TextProcessor()
    chunks = processor.split_text(raw_text)

    # 3. Vektörleştir ve Kaydet (Day 3)
    v_manager = VectorManager()
    vector_db = v_manager.create_vector_store(chunks)

    # --- TEST: SİSTEM GERÇEKTEN BULABİLİYOR MU? ---
    query = "Matplotlib nedir?" # PDF içeriğine uygun bir soru sor
    docs = vector_db.similarity_search(query, k=2) # En yakın 2 parçayı getir

    print("\n--- Arama Sonucu ---")
    for i, doc in enumerate(docs):
        print(f"\nİlgili Parça {i+1}:")
        print(doc.page_content[:200] + "...")

if __name__ == "__main__":
    main()