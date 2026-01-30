import os
import re
import uuid
import urllib
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader

load_dotenv()

def get_child_urls(parent_url: str, max_depth: int = 1000):

    def bs4_extractor(html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        return re.sub(r"\n\n+", "\n\n", soup.text).strip()

    pdf_urls = []
    html_urls = []

    try:
        loader = RecursiveUrlLoader(url=parent_url, max_depth=max_depth, extractor=bs4_extractor,continue_on_failure=True, check_response_status=True, autoset_encoding=True)
        documents = loader.lazy_load()

        for doc in tqdm(documents):
            raw_url = doc.metadata.get('source', '')
            url = urllib.parse.quote(raw_url, safe=':/')  # Encode spaces etc.

            try:
                # Use HEAD for fast check
                response = requests.head(url, allow_redirects=True, timeout=5)
                content_type = response.headers.get('Content-Type', '').lower()

                if response.status_code == 200:
                    if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
                        print(f"[PDF]  {url}")
                        pdf_urls.append(url)
                    elif 'text/html' in content_type:
                        print(f"[HTML] {url}")
                        html_urls.append(url)

            except requests.RequestException:
                continue  # Skip unreachable URLs

    except Exception as e:
        print(f"Unexpected error: {e}")

    return html_urls, pdf_urls

def save_urls_to_text_file(html_urls, pdf_urls: list, directory_name):
    with open(f"pdf_urls_{directory_name}.txt", "w", encoding="utf-8") as f:
        for url in pdf_urls:
            f.write(url + "\n")

    with open(f"html_urls_{directory_name}.txt", "w", encoding="utf-8") as f:
        for url in html_urls:
            f.write(url + "\n")

    return f"html_urls_{directory_name}.txt", f"pdf_urls_{directory_name}.txt"

def split_and_embed_urls_documents(html_url_txt, pdf_url_txt):

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_docs = []

    # ---- Load HTML URLs ----
    with open(f"{html_url_txt}", "r") as f:
        html_urls = [line.strip() for line in f if line.strip()]

    for url in html_urls:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            docs = text_splitter.split_documents(docs)
            all_docs.extend(docs)
            print(f"✅ Added HTML: {url}")
        except Exception as e:
            print(f"❌ Failed HTML: {url} - {e}")

    # ---- Load PDF URLs ----
    with open(f"{pdf_url_txt}", "r") as f:
        pdf_urls = [line.strip() for line in f if line.strip()]

    pdf_dir = "downloaded_pdfs"
    os.makedirs(pdf_dir, exist_ok=True)

    for url in pdf_urls:
        try:
            # Generate a unique filename
            safe_url = urllib.parse.quote(url, safe=':/') 
            filename = f"{uuid.uuid4().hex}.pdf"
            pdf_path = os.path.join(pdf_dir, filename)

            if not os.path.exists(pdf_path):  # avoid re-downloads
                response = requests.get(safe_url, timeout=15)
                response.raise_for_status()
                with open(pdf_path, "wb") as f:
                    f.write(response.content)
            
            # Download the PDF
            response = requests.get(url)
            response.raise_for_status()
            with open(pdf_path, "wb") as f:
                f.write(response.content)

            # Load and split PDF
            loader = PyMuPDFLoader(pdf_path)
            docs = loader.load()
            docs = text_splitter.split_documents(docs)
            all_docs.extend(docs)

            print(f"✅ Added PDF: {url}")
        except Exception as e:
            print(f"❌ Failed PDF: {url} - {e}")

    # ---- Build FAISS Index ----
    if all_docs:
        vectordb = FAISS.from_documents(all_docs, embedding_model)
        
        vectordb.save_local("embeddings_db")
        print("📦 FAISS vector DB stored with embeddings.")
    else:
        print("⚠️ No documents were added to FAISS.")

if __name__ == "__main__":
    parent_url = "https://www.svnit.ac.in/"
    html_urls, pdf_urls = get_child_urls(parent_url)

    print("Discovered URLs:")
    print(f"HTML URLs found: {len(html_urls)}")
    print(f"PDF URLs found: {len(pdf_urls)}")

    # 👉 Save immediately
    html_file, pdf_file = save_urls_to_text_file(html_urls, pdf_urls, "svnit")
    split_and_embed_urls_documents(html_file, pdf_file)

    print(f"✅ HTML URLs saved to {html_file}")
    print(f"✅ PDF URLs saved to {pdf_file}")

