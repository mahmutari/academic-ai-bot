from src.vector_store import VectorManager
from src.llm_manager import ChatManager

def main():
    # Başlangıç kurulumları
    v_manager = VectorManager()
    chat_manager = ChatManager()
    vector_db = v_manager.get_vector_store()

    print("\n" + "="*40)
    print("🎓 ACADEMIC ASSISTANT V1.0 HAZIR!")
    print("Çıkmak için 'exit' veya 'quit' yazabilirsin.")
    print("="*40 + "\n")

    while True:
        user_query = input("Siz: ")
        
        if user_query.lower() in ["exit", "quit", "q"]:
            print("Görüşmek üzere!")
            break
            
        print("Asistan düşünüyor...")
        
        # Dökümandan ilgili parçaları bul
        relevant_docs = vector_db.similarity_search(user_query, k=3)
        
        # Cevabı üret (Artık hafızalı ve hibrit!)
        answer = chat_manager.answer_question(user_query, relevant_docs)

        print(f"\nAI: {answer}\n")
        print("-" * 20)

if __name__ == "__main__":
    main()