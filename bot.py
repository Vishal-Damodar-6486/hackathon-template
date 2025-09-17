import os
from dotenv import load_dotenv
import streamlit as st

from langchain.schema import AIMessage, HumanMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.callbacks.base import BaseCallbackHandler

load_dotenv()

# -------------------------
# Custom Streaming Callback Handler
# -------------------------
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.text = ""
    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.placeholder.markdown(self.text)
    def on_llm_end(self, response, **kwargs):
        pass

# -------------------------
# Helper Functions (PDF, Embeddings, Vector Store, Context)
# -------------------------
def extract_text(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

def get_embeddings():
    return AzureOpenAIEmbeddings(
        model=os.getenv("AZURE_LLMsOpenAI_EMBEDDINGS_DEPLOYMENT_NAME"),
        azure_endpoint=os.getenv("AZURE_LLMsOpenAI_ENDPOINT"),
        api_key=os.getenv("AZURE_LLMsOpenAI_API_KEY"),
        openai_api_version=os.getenv("AZURE_LLMsOpenAI_API_VERSION")
    )

def get_vector_store(docs):
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(documents=docs, embedding=embeddings)
    vector_store.save_local('faiss_index')

def get_vector_store_retriever():
    embeddings = get_embeddings()
    try:
        db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
        return db.as_retriever(search_type="similarity", search_kwargs={"k": 20}, verbose=True)
    except Exception:
        return None

def get_context(question):
    retriever = get_vector_store_retriever()
    if retriever:
        docs = retriever.get_relevant_documents(question)
        print(docs)
        return "\n".join([doc.page_content for doc in docs])
    return ""

# -------------------------
# Build the Conversational Chain 
# -------------------------
def get_conversational_chain(callbacks=None):
    prompt = ChatPromptTemplate.from_template(
    """
    You are an AI chatbot specialized in analyzing RFQs.
    Answer only based on the retrieved context—no assumptions or hallucinations.

    *Chat History:*
    {chat_history}

    *RFQ Analysis Context:*
    {context}

    **Question:**
    {question}

    **Result:**
            """
    )
    output_parser = StrOutputParser()
    llm = AzureChatOpenAI(
        openai_api_key=os.getenv("AZURE_LLMsOpenAI_API_KEY"),
        openai_api_version=os.getenv("AZURE_LLMsOpenAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_LLMsOpenAI_GPT4O_DEPLOYMENT_NAME"),
        azure_endpoint=os.getenv("AZURE_LLMsOpenAI_ENDPOINT"),
        temperature=0.2,
        top_p=0.7,
        max_tokens=4096,
        streaming=True,
        callbacks=callbacks
    )
    chain = prompt | llm | output_parser
    return chain

# -------------------------
# Use session_state to persist chat history
# -------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Each entry is a tuple (role, message)

def render_chat_history():
    # Render each message once, in order.
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)

# -------------------------
# Process the user input: call LLM once to stream response
# -------------------------
def process_user_input(user_question, placeholder):
    context = get_context(user_question)
    # Build a simple string for chat history for the prompt.
    chat_history_str = "\n".join([f"{role}: {msg}" for role, msg in st.session_state.chat_history])
    callback_handler = StreamlitCallbackHandler(placeholder)
    chain = get_conversational_chain(callbacks=[callback_handler])
    inputs = {"question": user_question, "chat_history": chat_history_str, "context": context}
    response = chain.invoke(inputs)
    return response

# -------------------------
# Main Application
# -------------------------
def main():
    st.set_page_config(page_title="RFQ Chatbot", layout="wide")
    st.title("AI Chatbot")
    st.header("Summarize and Analyze your RFQs")
    
    # Sidebar: PDF Upload & Processing
    with st.sidebar:
        st.title("Upload RFQs:")
        pdf_file = st.file_uploader("Upload your PDF files", type="pdf")
        if st.button("Submit & Process"):
            if pdf_file:
                with st.spinner("Processing PDF..."):
                    temp_dir = "temp"
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_file_path = os.path.join(temp_dir, pdf_file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(pdf_file.getbuffer())
                    docs = extract_text(temp_file_path)
                    get_vector_store(docs)
                    st.session_state.pdf = True
                    st.success("PDF processed successfully!")
            else:
                st.warning("Please upload a PDF file.")
    
    # Render previous chat history (if any)
    render_chat_history()
    
    # Chat input area
    user_question = st.chat_input("Ask a question from PDF files")
    if user_question:
        if not st.session_state.get("pdf", False):
            st.warning("Please upload a PDF file first!")
            return
        
        # Step 1: Immediately display and store the user's query.
        st.session_state.chat_history.append(("user", user_question))
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Step 2: Create an assistant message block with a placeholder.
        with st.chat_message("assistant"):
            ai_placeholder = st.empty()
            ai_placeholder.markdown("Processing...")
            # Step 3: Invoke the LLM once to stream its output into the placeholder.
            final_response = process_user_input(user_question, ai_placeholder)
            ai_placeholder.markdown(final_response)
            # Append the assistant response to chat history.
            st.session_state.chat_history.append(("assistant", final_response))

if __name__ == "__main__":
    main()