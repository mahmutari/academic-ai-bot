from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

class ChatManager:
    def __init__(self, model_name="llama3.2"):
        self.llm = OllamaLLM(model=model_name)
        # Yapay zekaya nasıl davranması gerektiğini söylüyoruz (Prompt Engineering)
        self.template = """
        Sen akademik bir asistansın. Aşağıdaki bağlam (context) bilgilerini kullanarak kullanıcıya cevap ver.
        Eğer cevabı bağlamda bulamıyorsan, uydurma; sadece bilmediğini söyle.
        
        Bağlam: {context}
        Soru: {question}
        
        Cevap:
        """
        self.prompt = ChatPromptTemplate.from_template(self.template)

    def answer_question(self, question, retrieved_docs):
        # Gelen dökümanları tek bir metin haline getiriyoruz
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Prompt'u hazırlayıp LLM'e gönderiyoruz
        chain = self.prompt | self.llm
        response = chain.invoke({"context": context_text, "question": question})
        return response