from pandasai.llm.local_llm import LocalLLM
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
import base64

# Create basic auth header
def get_auth_header():
    credentials = base64.b64encode(b"root:eZfLK3X4-SX0i-UmgUBe6E").decode()
    return {"Authorization": f"Basic {credentials}"}

# Function to chat with CSV data
def chat_with_csv(df, query):
    # Initialize LocalLLM with cloud Ollama URL and authentication
    llm = LocalLLM(
        api_base="https://theaisource-u29564.vm.elestio.app:57987/v1",
        model="llama3",
        headers=get_auth_header()
    )
    # Initialize SmartDataframe with DataFrame and LLM configuration
    pandas_ai = SmartDataframe(df, config={"llm": llm})
    # Chat with the DataFrame using the provided query
    result = pandas_ai.chat(query)
    return result

# Set layout configuration for the Streamlit page
st.set_page_config(layout='wide')
# Set title for the Streamlit application
st.title("Multiple-CSV ChatApp powered by LLM")

# Upload multiple CSV files
input_csvs = st.sidebar.file_uploader("Upload your CSV files", type=['csv'], accept_multiple_files=True)

# Check if CSV files are uploaded
if input_csvs:
    # Select a CSV file from the uploaded files using a dropdown menu
    selected_file = st.selectbox("Select a CSV file", [file.name for file in input_csvs])
    selected_index = [file.name for file in input_csvs].index(selected_file)

    #load and display the selected csv file
    st.info("CSV uploaded successfully")
    data = pd.read_csv(input_csvs[selected_index])
    st.dataframe(data.head(3), use_container_width=True)

    #Enter the query for analysis
    st.info("Chat Below")
    input_text = st.text_area("Enter the query")

    #Perform analysis
    if input_text:
        if st.button("Chat with csv"):
            st.info("Your Query: " + input_text)
            result = chat_with_csv(data, input_text)
            st.success(result)
