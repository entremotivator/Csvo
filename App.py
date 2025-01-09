import streamlit as st
import pandas as pd
import base64
import request
from pandasai import SmartDataframe
from pandasai.llm.local_llm import LocalLLM

# Set page configuration
st.set_page_config(
    page_title="CSV Chat with Cloud Ollama",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# Ollama cloud configuration
OLLAMA_URL = "https://theaisource-u29564.vm.elestio.app:57987/v1"
OLLAMA_USERNAME = "root"
OLLAMA_PASSWORD = "eZfLK3X4-SX0i-UmgUBe6E"

def get_auth_header():
    """Generate basic auth header for Ollama API"""
    credentials = base64.b64encode(
        f"{OLLAMA_USERNAME}:{OLLAMA_PASSWORD}".encode()
    ).decode()
    return {"Authorization": f"Basic {credentials}"}

def initialize_llm():
    """Initialize the LLM with cloud Ollama configuration"""
    return LocalLLM(
        api_base=OLLAMA_URL,
        model="llama3.2",
        headers=get_auth_header()
    )

def process_query(dataframe, query):
    """Process a query against the dataframe using PandasAI"""
    try:
        llm = initialize_llm()
        pandas_ai = SmartDataframe(
            dataframe,
            config={
                "llm": llm,
                "verbose": True
            }
        )
        return pandas_ai.chat(query)
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return None

def display_data_info(df):
    """Display basic information about the loaded dataset"""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"Rows: {df.shape[0]}")
    with col2:
        st.info(f"Columns: {df.shape[1]}")
    with col3:
        st.info(f"Data Types: {len(df.dtypes.unique())}")

def main():
    # Application header
    st.title("📊 Intelligent CSV Analysis with Cloud Ollama")
    st.markdown("""
    Upload your CSV files and chat with your data using natural language queries.
    Powered by Llama 3 and PandasAI.
    """)
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_files = st.file_uploader(
            "Upload CSV files",
            type=['csv'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            selected_file = st.selectbox(
                "Select a CSV to analyze",
                [file.name for file in uploaded_files]
            )

    # Main content area
    if uploaded_files:
        # Load selected CSV
        selected_index = [file.name for file in uploaded_files].index(selected_file)
        df = pd.read_csv(uploaded_files[selected_index])
        
        # Display data preview
        st.header("🔍 Data Preview")
        display_data_info(df)
        st.dataframe(df.head(5), use_container_width=True)
        
        # Query section
        st.header("💬 Chat with your Data")
        query = st.text_area(
            "Enter your query about the data",
            height=100,
            placeholder="Example: What is the average of column X? or Show me a summary of the data."
        )
        
        if query:
            if st.button("🚀 Process Query", key="process"):
                with st.spinner("Processing your query..."):
                    result = process_query(df, query)
                    if result is not None:
                        st.success("Query processed successfully!")
                        st.write("### Results")
                        st.write(result)
    
    else:
        # Display welcome message when no file is uploaded
        st.info("👈 Please upload a CSV file to begin analysis")

if __name__ == "__main__":
    main()
