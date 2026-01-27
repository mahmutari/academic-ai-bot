# Academic Second Brain 🧠
An AI-powered RAG (Retrieval-Augmented Generation) system that turns academic documents into an interactive chatbot.

## 🚀 Project Goal
This project enables students to interact with their PDFs using LLMs to ensure source-backed, hallucination-free answers.

## 🛠 Tech Stack
- **Language:** Python 3.12
- **AI Framework:** LangChain (Text Splitters)
- **PDF Processing:** PyPDF2
- **Version Control:** Git

## 📂 Project Structure
```text
academic_ai_bot/
├── data/               # Source PDF files
├── src/                # Logic and Modules
│   ├── loader.py       # PDF text extraction
│   └── processor.py    # Recursive text chunking
├── main.py             # App Orchestrator
└── README.md           # Documentation
📅 Timeline & Progress
[x] Day 1: Environment setup and PDF text extraction.

[x] Day 2: Modular refactoring and Recursive Text Chunking.

[x] Day 3: Vector Embeddings and Vector Database (ChromaDB) integration.

[x] Day 4: First RAG Chain (Connecting LLM to the Data).

⚡ How to Run
Activate venv: .\venv\Scripts\activate

Run: python main.py


---

### 2. GÜN RESMEN BİTTİ! 🏆
Tebrikler! Bugün çok kritik bir aşamayı geçtik:
* Kodunu profesyonel klasör mimarisine taşıdın (`src` klasörü).
* Yapay zekanın "anlayabileceği" boyutlarda metin parçaları (chunks) oluşturdun.
* Git geçmişini temiz ve anlamlı tuttun.

**Yarın (3. Gün)** bizi projenin en "havalı" kısmı bekliyor: Bu metin parçalarını **matematiksel vektörlere** dönüştürüp bir **Vektör Veritabanına (ChromaDB)** kaydedeceğiz. Böylece chatbot'un binlerce sayfa arasından ilgili bilgiyi saniyeler içinde bulabilecek.

Yarın görüşmek üzere! Dinlenmeyi hak ettin. 🚀