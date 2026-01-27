from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory

class ChatManager:
    def __init__(self, model_name="llama3.2"):
        # Temperature=0.3: Tekrarı önlemek ve daha doğal konuşmak için idealdir.
        self.llm = OllamaLLM(model=model_name, temperature=0.3)
        self.history = ChatMessageHistory()
        
    # llm_manager.py içindeki template bölümü:

        self.template = """
### ROL
Sen, hem genel akademik bilgilere sahip bir öğretmen hem de teknik dökümanları analiz eden uzman bir asistansın.

### TALİMATLAR
1. ÖNCELİK KONTROLÜ: Kullanıcı dökümanla (Matplotlib, Spyder, grafikler) ilgili bir şey soruyorsa, mutlaka BAĞLAM (Context) içindeki teknik bilgileri kullan.
2. DOĞRULAMA: Eğer bilgi bağlamda VARSA, sakın "Dökümanda yok" deme. Bilgiyi dökümandan aldığını belirterek açıkla.
3. GENEL BİLGİ: Eğer soru dökümanda bulunmayan tamamen farklı bir akademik konuysa (örneğin "sıfatlar", "zamirler"), kendi genel bilgilerini kullanarak detaylı ve samimi bir açıklama yap.
4. ÜSLUP: Samimi, akademik ve net ol. Gereksiz tekrarlardan kaçın.

BAĞLAM: {context}
GEÇMİŞ: {chat_history}
SORU: {question}

CEVAP:
"""
        self.prompt = ChatPromptTemplate.from_template(self.template)

    def _format_history(self):
        return "\n".join([f"{'İnsan' if m.type=='human' else 'AI'}: {m.content}" for m in self.history.messages[-4:]])

    def answer_question(self, question, retrieved_docs):
        # 1. Bağlamı ve döküman alaka kontrolünü hazırla
        context_text = ""
        is_relevant = False
        
        if retrieved_docs:
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
            # Basit bir alaka kontrolü: Soru döküman anahtar kelimelerini içeriyor mu?
            keywords = ["plot", "matplotlib", "graph", "fig", "ax", "spyder", "çizim", "grafik"]
            if any(key in question.lower() for key in keywords):
                is_relevant = True

        # 2. Cevabı üret
        chain = self.prompt | self.llm
        response = chain.invoke({
            "context": context_text if is_relevant else "BU SORU İÇİN DÖKÜMAN KULLANMA.", 
            "chat_history": self._format_history(),
            "question": question
        })
        
        # 3. Kaynak Gösterme Kararı
        final_response = response
        if is_relevant and retrieved_docs:
            pages = sorted(list(set([doc.metadata.get('page', 0) + 1 for doc in retrieved_docs])))
            final_response = f"{response}\n\n📍 (Kaynak: Sayfa {', '.join(map(str, pages))})"
        
        # 4. Geçmişi kaydet
        self.history.add_user_message(question)
        self.history.add_ai_message(final_response)
        return final_response