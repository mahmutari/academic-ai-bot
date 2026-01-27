import streamlit as st
from streamlit_chat import message
import os
from src.loader import PDFLoader
from src.processor import TextProcessor
from src.vector_store import VectorManager
from src.llm_manager import ChatManager

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Academic Second Brain", page_icon="🎓", layout="wide")

# --- BİLEŞENLERİ BAŞLATMA ---
# Vektör veritabanı artık RAM'de olduğu için doğrudan session_state içinde tutuyoruz.
if "v_manager" not in st.session_state:
    st.session_state.v_manager = VectorManager()
if "chat_manager" not in st.session_state:
    st.session_state.chat_manager = ChatManager()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# İşlemci araçları
loader = PDFLoader()
processor = TextProcessor()

# --- YARDIMCI FONKSİYONLAR ---
def clear_everything():
    """Tüm sistemi ve RAM'deki veritabanını sıfırlar."""
    st.session_state.vector_db = None
    st.session_state.messages = []
    st.session_state.chat_manager.history.clear()
    st.success("Tüm sistem ve hafıza temizlendi!")

# --- SIDEBAR (YAN PANEL) ---
with st.sidebar:
    st.title("📂 Dosya Yönetimi")
    uploaded_file = st.file_uploader("Analiz edilecek PDF'i seçin", type="pdf")
    
    if uploaded_file:
        if st.button("🚀 Dökümanı İşle ve RAM'e Yükle"):
            with st.spinner("Eski veriler temizleniyor ve döküman analiz ediliyor..."):
                # 1. Geçici dosyayı oluştur (Okuyucu için)
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # 2. PDF'i işle ve parçala
                raw_docs = loader.get_pdf_documents("temp.pdf")
                chunks = processor.split_docs(raw_docs)
                
                # 3. YENİ: Vektör veritabanını oluştur ve doğrudan RAM'e (session_state) ata
                # Bu işlem, eski 'vector_db' nesnesini otomatik olarak bellekten düşürür.
                st.session_state.vector_db = st.session_state.v_manager.create_vector_store(chunks)
                
                # 4. Sohbet geçmişini taze başlangıç için temizle
                st.session_state.messages = []
                st.session_state.chat_manager.history.clear()
                
                st.success("Döküman başarıyla RAM'e yüklendi! Artık soru sorabilirsiniz.")
                st.rerun()

    st.divider()
    if st.button("🔴 Sistemi Tamamen Sıfırla"):
        clear_everything()
        st.rerun()

# --- ANA EKRAN (CHAT ARAYÜZÜ) ---
st.title("🎓 Academic Second Brain")
st.caption("RAM üzerinde çalışan hibrit akademik asistan (Hızlı ve Güvenli).")

# Sohbet Geçmişini Ekranda Göster
for i, msg in enumerate(st.session_state.messages):
    message(msg["content"], is_user=msg["is_user"], key=f"msg_{i}")

# Kullanıcı Girişi
user_input = st.chat_input("Sorunuzu buraya yazın...")

if user_input:
    # 1. Kullanıcı mesajını listeye ekle
    st.session_state.messages.append({"content": user_input, "is_user": True})
    
    with st.spinner("Asistan düşünüyor..."):
        relevant_docs = []
        
        # 2. Eğer RAM'de bir veritabanı yüklüyse arama yap
        if st.session_state.vector_db is not None:
            relevant_docs = st.session_state.vector_db.similarity_search(user_input, k=3)
        
        # 3. Yanıt Üret (relevant_docs boşsa ChatManager genel bilgisiyle cevap verir)
        response = st.session_state.chat_manager.answer_question(user_input, relevant_docs)
        
        # 4. Yanıtı mesajlara ekle
        st.session_state.messages.append({"content": response, "is_user": False})
        st.rerun()