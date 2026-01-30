# 🤖 SVNIT CSE Chatbot (RAG-based)

An intelligent **Retrieval-Augmented Generation (RAG)** chatbot designed to answer queries related to the **Computer Science & Engineering Department of SVNIT, Surat**. The system crawls official SVNIT web pages and PDFs, builds semantic embeddings, and serves accurate, context-grounded answers through an interactive **Streamlit** interface.

---

## 🚀 What This Project Does

* 🔎 Crawls SVNIT official website recursively
* 🌐 Extracts **HTML pages** and **PDF documents**
* ✂️ Splits content into semantic chunks
* 🧠 Converts text into vector embeddings using **Sentence Transformers**
* 📦 Stores embeddings in a **FAISS vector database**
* 💬 Uses an LLM (via Hugging Face endpoint) to answer user questions
* 🖥️ Provides a clean, chat-style UI using **Streamlit**

This ensures **low hallucination**, **source-grounded answers**, and **department-specific accuracy**.


## 🧠 Architecture (High-Level)

```
User Question
     ↓
Streamlit UI (app.py)
     ↓
FAISS Retriever (Top-K similar chunks)
     ↓
Context + Chat History
     ↓
LLM (HuggingFaceEndpoint)
     ↓
Final Answer
```



## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/thenigma/CSE-Chatbot.git
cd CSE-Chatbot
```

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## 🔐 Environment Variables

Create a `.env` file in the root directory:

```env
HUGGINGFACEHUB_API_TOKEN=your_api_key_here
```

> ⚠️ `.env` is ignored by Git for security reasons.


## 🧱 Building the Vector Database

Run the helper script **once** to crawl SVNIT data and build embeddings:

```bash
python helper.py
```

This will:

* Crawl `https://www.svnit.ac.in/`
* Identify HTML & PDF resources
* Generate embeddings
* Store them in `embeddings_db/`


## ▶️ Running the Chatbot

After embeddings are created:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.


## 🧪 Example Queries

* "Who is the HOD of CSE department at SVNIT?"
* "Tell me about SVNIT admission process"

If the answer is **not present in context**, the chatbot will safely respond:

> *"I don't know."*


## 🛠️ Tech Stack

* **Python**
* **LangChain**
* **Hugging Face Transformers**
* **FAISS** (Vector Store)
* **Streamlit** (UI)
* **BeautifulSoup + Requests** (Web Crawling)


## 📌 Future Improvements

* 🔗 Source citation in answers
* 📄 Per-document metadata filtering
* 🔄 Incremental re-indexing
* 🌍 Deployment on Hugging Face Spaces / Streamlit Cloud


⭐ If you find this project useful, feel free to star the repository!
