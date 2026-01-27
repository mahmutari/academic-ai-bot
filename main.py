from src.loader import PDFLoader
from src.processor import TextProcessor
from src.vector_store import VectorManager
from src.llm_manager import ChatManager

def main():
    # 1. Altyapıyı Hazırla
    v_manager = VectorManager()
    chat_manager = ChatManager()
    
    # 2. Veritabanını Yükle (Day 3'te oluşturmuştuk)
    vector_db = v_manager.get_vector_store()

    # 3. Kullanıcıdan Soru Al
    user_query = input("\nAkademik Asistanına bir soru sor: ")

    # 4. RAG Akışı (Retrieval - Augmented - Generation)
    print("Cevap hazırlanıyor...")
    relevant_docs = vector_db.similarity_search(user_query, k=3) # Bilgiyi bul (Retrieval)
    answer = chat_manager.answer_question(user_query, relevant_docs) # Cevabı üret (Generation)

    print("\n--- AI CEVABI ---")
    print(answer)

if __name__ == "__main__":
    main()