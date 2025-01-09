import streamlit as st
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
import json
import time
from urllib3.exceptions import InsecureRequestWarning
import logging
from typing import Optional, Dict, Any

# Suppress only the single InsecureRequestWarning
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
class OllamaConfig:
    BASE_URL = "https://theaisource-u29564.vm.elestio.app:57987"
    USERNAME = "root"
    PASSWORD = "eZfLK3X4-SX0i-UmgUBe6E"
    MODEL = "llama2"
    MAX_RETRIES = 3
    RETRY_DELAY = 2

class OllamaAPI:
    def __init__(self):
        self.auth = HTTPBasicAuth(OllamaConfig.USERNAME, OllamaConfig.PASSWORD)
        self.session = requests.Session()
        self.session.verify = False  # Required for self-signed certificates
        
    def _make_request(self, endpoint: str, data: Dict[str, Any], retries: int = 0) -> Optional[str]:
        """Make a request to the Ollama API with retry logic"""
        url = f"{OllamaConfig.BASE_URL}{endpoint}"
        headers = {"Content-Type": "application/json"}
        
        try:
            response = self.session.post(
                url,
                headers=headers,
                auth=self.auth,
                json=data,
                timeout=30
            )
            
            # Log response details for debugging
            logger.info(f"Status Code: {response.status_code}")
            logger.info(f"Response Headers: {response.headers}")
            
            if response.status_code == 200:
                return response.json().get('response', '')
            
            # Handle specific error cases
            elif response.status_code == 500:
                if retries < OllamaConfig.MAX_RETRIES:
                    time.sleep(OllamaConfig.RETRY_DELAY)
                    return self._make_request(endpoint, data, retries + 1)
                else:
                    st.error("Maximum retries reached. Server is not responding correctly.")
            elif response.status_code == 401:
                st.error("Authentication failed. Please check credentials.")
            elif response.status_code == 404:
                st.error("API endpoint not found. Please check the URL.")
            else:
                st.error(f"Unexpected error: Status code {response.status_code}")
            
            return None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            st.error(f"Connection error: {str(e)}")
            return None

    def generate_response(self, prompt: str) -> Optional[str]:
        """Generate a response using the Ollama API"""
        data = {
            "model": OllamaConfig.MODEL,
            "prompt": prompt,
            "stream": False
        }
        return self._make_request("/api/generate", data)

class DataAnalyzer:
    @staticmethod
    def create_analysis_prompt(df: pd.DataFrame, query: str) -> str:
        """Create a detailed prompt for data analysis"""
        # Get basic statistics
        numeric_stats = df.describe().to_string() if not df.empty else "No numeric data"
        
        # Get sample data (first 5 rows)
        sample_data = df.head().to_string() if not df.empty else "No data available"
        
        # Create a comprehensive prompt
        prompt = f"""
        As a data analyst, analyze the following dataset:

        Columns: {', '.join(df.columns)}
        Data Types: {df.dtypes.to_string()}
        
        Sample Data:
        {sample_data}
        
        Basic Statistics:
        {numeric_stats}
        
        User Question: {query}
        
        Please provide a detailed analysis answering the user's question.
        Format the response in a clear, easy-to-read manner.
        Include relevant statistics and insights.
        """
        return prompt.strip()

def main():
    st.title("📊 Advanced CSV Analyzer")
    st.markdown("""
    ### Upload your CSV file and get AI-powered insights
    This tool uses advanced language models to analyze your data and answer questions.
    """)
    
    # Initialize API client
    ollama_client = OllamaAPI()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            with st.spinner("Loading data..."):
                df = pd.read_csv(uploaded_file)
                
                # Display data info
                st.write("### Data Overview")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", df.shape[0])
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                    st.metric("Missing Values", df.isna().sum().sum())
                
                # Display sample data
                st.write("### Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Query section
                st.write("### Ask Questions About Your Data")
                query = st.text_area(
                    "What would you like to know?",
                    placeholder="Example: What are the main trends in this data? What is the average of column X?",
                    height=100
                )
                
                if query and st.button("Analyze Data", type="primary"):
                    with st.spinner("Analyzing your data..."):
                        # Create analysis prompt
                        prompt = DataAnalyzer.create_analysis_prompt(df, query)
                        
                        # Get response from API
                        response = ollama_client.generate_response(prompt)
                        
                        if response:
                            st.write("### Analysis Results")
                            st.write(response)
                            
                            # Add option to download results
                            st.download_button(
                                label="Download Analysis",
                                data=response,
                                file_name="analysis_results.txt",
                                mime="text/plain"
                            )
        
        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty. Please upload a valid CSV file.")
        except pd.errors.ParserError:
            st.error("Error parsing the CSV file. Please make sure it's properly formatted.")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            st.error(f"An unexpected error occurred: {str(e)}")
    else:
        st.info("👆 Please upload a CSV file to begin analysis")

if __name__ == "__main__":
    main()
