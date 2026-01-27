import os
from src.loader import PDFLoader
from src.processor import TextProcessor
from src.vector_store import VectorManager
from src.llm_manager import ChatManager

def main():
    # 1. Klasör ve Dosya Yolları
    pdf_path = "data/Python-plot.pdf" 
    db_path = "./db_academic"
    
    print("Sistem bileşenleri hazırlanıyor...")
    v_manager = VectorManager(db_directory=db_path)

    # 2. OTOMATİK KURULUM: Veritabanı yoksa PDF'i en baştan işle
    if not os.path.exists(db_path):
        print(f"⚠️ '{db_path}' bulunamadı. Sayfa numaralarıyla birlikte yeniden oluşturuluyor...")
        
        if not os.path.exists(pdf_path):
            print(f"❌ Hata: '{pdf_path}' bulunamadı! Lütfen PDF'i data klasörüne koyun.")
            return

        loader = PDFLoader()
        processor = TextProcessor()

        # PDF'i metadata (sayfa bilgisi) ile yükle ve parçala
        raw_docs = loader.get_pdf_documents(pdf_path)
        chunks = processor.split_docs(raw_docs)
        
        # Yeni 'from_documents' mantığıyla veritabanını oluştur
        v_manager.create_vector_store(chunks)
        print("✅ Veritabanı sayfa numaralarıyla birlikte başarıyla oluşturuldu!")

    # 3. Chat Arayüzünü Başlat
    try:
        chat_manager = ChatManager()
        vector_db = v_manager.get_vector_store()
        
        os.system('cls' if os.name == 'nt' else 'clear')
        print("="*50)
        print("      🎓 ACADEMIC SECOND BRAIN - Metadata Fixed")
        print("="*50 + "\n")

        while True:
            user_query = input("Siz > ")
            if user_query.lower() in ["exit", "quit", "q"]: break
            if not user_query.strip(): continue

            print("\n🔍 Kanıtlar toplanıyor...")
            relevant_docs = vector_db.similarity_search(user_query, k=3)
            answer = chat_manager.answer_question(user_query, relevant_docs)

            print(f"\nAI > {answer}\n")
            print("-" * 50)

    except Exception as e:
        print(f"❌ Çalışma hatası: {e}")

if __name__ == "__main__":
    main()