import streamlit as st
from pdfminer.high_level import extract_text
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import os
import httpx
import requests
from langsmith import traceable, Client
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.callbacks.base import BaseCallbackHandler
import httpx

# Create HTTP client with verify=False for LangChain
client = httpx.Client(verify=False)

# === Monkeypatch requests to disable SSL verification globally ===
for method in ("get", "post", "put", "delete", "head", "options", "patch"):
    original = getattr(requests, method)

    def insecure_request(*args, _original=original, **kwargs):
        kwargs["verify"] = False
        return _original(*args, **kwargs)

    setattr(requests, method, insecure_request)

# Set tiktoken cache dir
tiktoken_cache_dir = "./token"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGCHAIN_API_KEY"] = ""  # Replace with your key
os.environ["LANGCHAIN_PROJECT"] = "RAG PDF ChatBOT"

#  --------------------------
#   Session to bypass the ssl certificate error for accessing langsmith
# ------------------------------
Session=requests.Session()
Session.verify=False
openAI_client=Client(session=Session)

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
@traceable(name="Text Extraction",client=openAI_client)
def extract_text(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

@traceable(name="Embeddings",client=openAI_client)
def get_embeddings():
    return OpenAIEmbeddings(
        base_url="https://genailab.tcs.in",
        model="azure/genailab-maas-text-embedding-3-large",
        api_key="sk-g0RigALF05KUmlonLK3JHg",
        http_client=client,
        )

@traceable(name="Onetime DB creation",client=openAI_client)
def get_vector_store(docs):
    embeddings = get_embeddings()
    # vector_store = FAISS.from_documents(documents=docs, embedding=embeddings)
    # vector_store.save_local('faiss_index')
    vector_store = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory="./chroma_index")
    vector_store.persist()
    
@traceable(name="retriver tool",client=openAI_client)
def get_vector_store_retriever():
    try:
        # db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
        # Initialize the embedding model
        embedding_model = get_embeddings()

        # Load the persisted vector store
        db = Chroma(persist_directory="./chroma_index", embedding_function=embedding_model)
        return db.as_retriever(search_type="similarity", search_kwargs={"k": 20}, verbose=True)
    except Exception:
        return None

@traceable(name="context retriver",client=openAI_client)
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
    llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="azure_ai/genailab-maas-DeepSeek-V3-0324",
    api_key="sk-g0RigALF05KUmlonLK3JHg",
    http_client=client,
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
@traceable(name="OpenAI RAG Chatbot tracing with LangSmith",client=openAI_client)
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
