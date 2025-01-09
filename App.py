import streamlit as st
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.ollama import Ollama
import requests
from requests.auth import HTTPBasicAuth
from urllib3.exceptions import InsecureRequestWarning
import logging

# Suppress insecure request warnings
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="PandasAI CSV Analyzer",
    page_icon="📊",
    layout="wide"
)

# Ollama Configuration
OLLAMA_HOST = "https://theaisource-u29564.vm.elestio.app:57987"
OLLAMA_USERNAME = "root"
OLLAMA_PASSWORD = "eZfLK3X4-SX0i-UmgUBe6E"

class CustomOllama(Ollama):
    def __init__(self):
        super().__init__(
            model="llama2",
            base_url=OLLAMA_HOST,
            auth=HTTPBasicAuth(OLLAMA_USERNAME, OLLAMA_PASSWORD)
        )
        self.session = requests.Session()
        self.session.verify = False
        self.session.auth = HTTPBasicAuth(OLLAMA_USERNAME, OLLAMA_PASSWORD)

    def _generate(self, prompt, stream=False):
        try:
            headers = {"Content-Type": "application/json"}
            data = {
                "model": self.model,
                "prompt": prompt,
                "stream": stream
            }
            
            response = self.session.post(
                f"{self.base_url}/api/generate",
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                logger.error(f"API Error: {response.status_code}, {response.text}")
                return f"Error: {response.status_code}"
                
        except Exception as e:
            logger.error(f"Request error: {str(e)}")
            return f"Error: {str(e)}"

def initialize_pandasai():
    """Initialize PandasAI with custom Ollama configuration"""
    try:
        llm = CustomOllama()
        pandas_ai = PandasAI(llm)
        return pandas_ai
    except Exception as e:
        st.error(f"Error initializing PandasAI: {str(e)}")
        return None

def analyze_data(pandas_ai, df, query):
    """Analyze data using PandasAI"""
    try:
        return pandas_ai.run(df, prompt=query)
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        st.error(f"Error during analysis: {str(e)}")
        return None

def main():
    st.title("📊 PandasAI CSV Analyzer")
    st.markdown("Analyze your CSV data using AI-powered insights")

    # Initialize PandasAI
    pandas_ai = initialize_pandasai()
    if not pandas_ai:
        st.error("Failed to initialize PandasAI. Please check the configuration.")
        return

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Load data
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

            # Show data preview
            st.write("### Data Preview")
            st.dataframe(df.head(), use_container_width=True)

            # Query section
            st.write("### Ask Questions")
            query = st.text_area(
                "What would you like to know about the data?",
                placeholder="Example: What are the trends in this data? What is the average of column X?"
            )

            if query and st.button("Analyze", type="primary"):
                with st.spinner("Analyzing your data..."):
                    # Test API connection
                    try:
                        test_response = requests.get(
                            f"{OLLAMA_HOST}/api/tags",
                            auth=HTTPBasicAuth(OLLAMA_USERNAME, OLLAMA_PASSWORD),
                            verify=False,
                            timeout=5
                        )
                        
                        if test_response.status_code == 200:
                            # Proceed with analysis
                            result = analyze_data(pandas_ai, df, query)
                            if result is not None:
                                st.write("### Analysis Results")
                                st.write(result)
                        else:
                            st.error(f"API connection test failed: {test_response.status_code}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Connection error: {str(e)}")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            logger.error(f"Processing error: {str(e)}")
    else:
        st.info("👆 Please upload a CSV file to begin")

if __name__ == "__main__":
    main()
