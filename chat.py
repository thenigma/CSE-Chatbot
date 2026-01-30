import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import warnings
import os

warnings.filterwarnings("ignore")
load_dotenv()

st.set_page_config(page_title="SVNIT Chatbot", page_icon="🤖")
st.title("🤖 SVNIT Chatbot")
st.markdown("###### Hello I'm your helpful chatbot assistant for Computer Science Engineering Department of SVNIT, Surat.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def load_chain():
    # embeddings + retriever
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.load_local("embeddings_db", embeddings=embedding, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 25})

    # modern HuggingFaceHub wrapper (uses InferenceClient under the hood)
    llm = HuggingFaceHub(
        repo_id="meta-llama/Llama-2-7b-chat-hf",
        model_kwargs={
            "temperature": 0.7,
            "max_new_tokens": 512
        }
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Your  are a helpful chatbot assistant with expertise in extracting and answering questions of SVNIT, Surat. "
                   "Use the provided context and chat history to answer the user's query. "
                   "If the answer is not found, respond with 'I don't know.' You may also suggest a source of information if it is not available in context. Do not hallucinate."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    parser = StrOutputParser()
    chain = prompt | llm | parser

    return retriever, chain

retriever, chain = load_chain()

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message("user" if isinstance(msg, HumanMessage) else "ai"):
        st.markdown(msg.content)

user_input = st.chat_input("Ask me something about SVNIT...")

if user_input:
    st.chat_message("user").markdown(user_input)

    docs = retriever.invoke(user_input)
    context = "\n\n".join([doc.page_content for doc in docs])

    response = chain.invoke({
        "context": context,
        "question": user_input,
        "chat_history": st.session_state.chat_history
    })

    st.chat_message("ai").markdown(response)

    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=response))
