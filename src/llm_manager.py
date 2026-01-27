from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory

class ChatManager:
    def __init__(self, model_name="llama3.2"):
        self.llm = OllamaLLM(model=model_name)
        
        # 1. Ham Mesaj Geçmişi: Listeyi burada tutuyoruz
        self.history = ChatMessageHistory()
        
        self.template = """
        Sen akademik ve samimi bir asistansın. 
        Sana verilen bağlam (context) döküman bilgilerini ve geçmiş sohbeti (chat_history) kullanarak cevap ver.
        
        KURAL 1: Soru dökümanla ilgiliyse dökümandaki bilgiyi temel al.
        KURAL 2: Soru genel bir konuysa (Selam, nasılsın vb.) kendi genel bilgini kullanarak nazikçe cevap ver.
        KURAL 3: Cevap dökümandan geliyorsa bunu belirt.
        
        Bağlam: {context}
        Geçmiş Sohbet: {chat_history}
        Kullanıcı Sorusu: {question}
        
        Cevap:
        """
        self.prompt = ChatPromptTemplate.from_template(self.template)

    def _format_history(self):
        """Geçmiş mesajları metin bloğuna çevirir (LLM'in anlaması için)."""
        formatted_text = ""
        for msg in self.history.messages:
            prefix = "İnsan: " if msg.type == "human" else "AI: "
            formatted_text += f"{prefix}{msg.content}\n"
        return formatted_text

    def answer_question(self, question, retrieved_docs):
        # Döküman parçalarını birleştir
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Sohbet geçmişini metin formatında al
        chat_history_text = self._format_history()
        
        # Zinciri çalıştır
        chain = self.prompt | self.llm
        response = chain.invoke({
            "context": context_text, 
            "chat_history": chat_history_text,
            "question": question
        })
        
        # 2. ÖNEMLİ: Mesajları geçmişe ekle
        self.history.add_user_message(question)
        self.history.add_ai_message(response)
        
        return response