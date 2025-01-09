import streamlit as st
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
import json
import time
from urllib3.exceptions import InsecureRequestWarning
import logging

# Suppress insecure request warnings
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="CSV Analyzer",
    page_icon="📊",
    layout="wide"
)

# API Configuration
OLLAMA_BASE_URL = "https://theaisource-u29564.vm.elestio.app:57987"
AUTH = HTTPBasicAuth("root", "eZfLK3X4-SX0i-UmgUBe6E")

def query_ollama(prompt: str) -> str:
    """
    Send a query to Ollama API with proper error handling
    """
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "llama2",
        "prompt": prompt
    }
    
    try:
        # First try to check if the server is responding
        health_check = requests.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            auth=AUTH,
            verify=False,
            timeout=10
        )
        
        if health_check.status_code != 200:
            st.error(f"Server health check failed with status code: {health_check.status_code}")
            return None
            
        # Make the actual query
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            headers=headers,
            auth=AUTH,
            json=data,
            verify=False,
            timeout=30
        )
        
        if response.status_code == 200:
            try:
                result = response.json()
                return result.get('response', 'No response data received')
            except json.JSONDecodeError:
                st.error("Failed to decode JSON response from server")
                return None
                
        else:
            st.error(f"Server returned error code: {response.status_code}")
            logger.error(f"Response content: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        logger.error(f"Request error: {str(e)}")
        return None

def analyze_data(df: pd.DataFrame, query: str) -> str:
    """
    Prepare data analysis prompt and get response from Ollama
    """
    # Create a concise context about the data
    data_info = (
        f"Data Overview:\n"
        f"- Total rows: {df.shape[0]}\n"
        f"- Columns: {', '.join(df.columns)}\n"
        f"- Sample data:\n{df.head(3).to_string()}\n\n"
        f"User Question: {query}\n\n"
        "Please analyze this data and answer the question with clear explanations."
    )
    
    return query_ollama(data_info)

def main():
    st.title("📊 CSV Data Analyzer")
    st.markdown("Upload your CSV file and ask questions about your data")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            # Show data preview
            st.write("### Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Basic stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isna().sum().sum())
            
            # Query section
            st.write("### Ask Questions")
            query = st.text_area(
                "What would you like to know about the data?",
                placeholder="Example: What are the main trends? What is the average of column X?"
            )
            
            if query and st.button("Analyze", type="primary"):
                with st.spinner("Analyzing your data..."):
                    # Test API connection first
                    try:
                        test_response = requests.get(
                            f"{OLLAMA_BASE_URL}/api/tags",
                            auth=AUTH,
                            verify=False,
                            timeout=5
                        )
                        if test_response.status_code == 200:
                            # Proceed with analysis
                            result = analyze_data(df, query)
                            if result:
                                st.write("### Analysis Results")
                                st.write(result)
                        else:
                            st.error("Unable to connect to the API server. Please try again later.")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Connection test failed: {str(e)}")
                        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            logger.error(f"Processing error: {str(e)}")
    else:
        st.info("👆 Please upload a CSV file to begin")

if __name__ == "__main__":
    main()
