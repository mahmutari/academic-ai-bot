from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory

class ChatManager:
    def __init__(self, model_name="llama3.2"):
        # Temperature=0: Modeli daha 'ciddi' ve 'uydurmayan' hale getirir.
        self.llm = OllamaLLM(model=model_name, temperature=0)
        self.history = ChatMessageHistory()
        
        # Daha katı bir talimat seti
        self.template = """
        Sen bir akademik analiz robotusun. Sadece dökümana sadık kal.
        
        KURALLAR:
        1. Soru dökümanla ilgiliyse, SADECE dökümandaki teknik terimleri kullan.
        2. Bilgi dökümanda yoksa, asla kendi bilgini ekleme; 'Bu bilgi dökümanda bulunmuyor' de.
        3. Matematiksel formülleri açık ve net yaz.
        
        BAĞLAM (Döküman): {context}
        GEÇMİŞ: {chat_history}
        SORU: {question}
        
        CEVAP:
        """
        self.prompt = ChatPromptTemplate.from_template(self.template)

    def _format_history(self):
        return "\n".join([f"{'İnsan' if m.type=='human' else 'AI'}: {m.content}" for m in self.history.messages[-6:]])

    def answer_question(self, question, retrieved_docs):
        # Metadata Kontrolü: doc.metadata içinde 'page' yoksa 0 ata
        page_numbers = []
        for doc in retrieved_docs:
            p = doc.metadata.get('page')
            if p is not None:
                page_numbers.append(p + 1)
        
        pages_set = sorted(list(set(page_numbers)))
        source_info = f"Sayfa {', '.join(map(str, pages_set))}" if pages_set else "Bilinmiyor"

        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        chain = self.prompt | self.llm
        response = chain.invoke({
            "context": context_text, 
            "chat_history": self._format_history(),
            "question": question
        })
        
        final_response = f"{response}\n\n📍 (Kaynak: {source_info})"
        
        self.history.add_user_message(question)
        self.history.add_ai_message(final_response)
        return final_response