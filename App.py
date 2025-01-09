import streamlit as st
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
import json

# Page configuration
st.set_page_config(
    page_title="CSV Chat with Ollama",
    page_icon="📊",
    layout="wide"
)

# Ollama API configuration
OLLAMA_URL = "https://theaisource-u29564.vm.elestio.app:57987"
AUTH = HTTPBasicAuth("root", "eZfLK3X4-SX0i-UmgUBe6E")

def query_ollama(prompt, model="llama2"):
    """Send a query to Ollama API"""
    try:
        headers = {"Content-Type": "application/json"}
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            headers=headers,
            auth=AUTH,
            json=data,
            verify=False  # Only if needed for self-signed certificates
        )
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error querying Ollama: {str(e)}")
        return None

def analyze_data(df, query):
    """Analyze the dataframe based on the query"""
    # Create a context about the data
    context = f"""
    Analyze this DataFrame with columns: {', '.join(df.columns)}
    First few rows: {df.head().to_string()}
    Summary statistics: {df.describe().to_string()}
    
    User Question: {query}
    
    Provide a clear and concise analysis based on the data.
    """
    
    return query_ollama(context)

def main():
    st.title("📊 CSV Analysis with Ollama")
    st.markdown("Upload your CSV file and ask questions about your data.")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Load and display data
            df = pd.read_csv(uploaded_file)
            st.write("### Data Preview")
            st.dataframe(df.head())
            
            # Display basic statistics
            st.write("### Basic Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"Rows: {df.shape[0]}")
            with col2:
                st.info(f"Columns: {df.shape[1]}")
            with col3:
                st.info(f"Data Types: {len(df.dtypes.unique())}")
            
            # Query input
            st.write("### Ask Questions")
            query = st.text_area(
                "What would you like to know about the data?",
                placeholder="Example: What is the average of column X? What are the main trends?"
            )
            
            if query and st.button("Analyze"):
                with st.spinner("Analyzing your data..."):
                    result = analyze_data(df, query)
                    if result:
                        st.write("### Analysis Results")
                        st.write(result)
                        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        st.info("👆 Please upload a CSV file to begin")

if __name__ == "__main__":
    main()
