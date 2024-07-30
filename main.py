import numpy as np
import pandas as pd
from getpass import getpass
import streamlit as st
import os
import langchain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from langchain.docstore.document import Document
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import openai
import base64
from PIL import Image
from pdf2image import convert_from_path
from utils import *
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

# Store openAI key and create a client 
openai.api_key = os.getenv('OPENAI_API_KEY')
client = openai.OpenAI()

# Generate the context from the PDF file using the client
df_info = context_generation(client)

# Load the information from the dataframe
loader = DataFrameLoader(df_info, page_content_column="sumarization")
docs = loader.load()

# Create a FAISS vector database to store the embeddings and the summary
faiss_index = FAISS.from_documents(docs, OpenAIEmbeddings())

# Define basic model of OpenAI as LLM
chat = ChatOpenAI(model_name='gpt-3.5-turbo')

# Read the PDF file to show the customer the option to download the catalog
with open("./Bruno_child_offers.pdf", "rb") as pdf_file:
    PDFbyte = pdf_file.read()
    
# Define basic configuration of web page
st.set_page_config(page_title='Konecta Technical test Nestor Calvo', page_icon=':shark:', layout='wide')

# If no history of chats, the define an empty array 
if 'history' not in st.session_state:
    st.session_state.history = []

if 'message_chain' not in st.session_state:
    st.session_state.message_chain = []

st.markdown('# Konecta Technical test Nestor Calvo')
st.title('ChatBot')

# Shows previous message if any
for message in st.session_state.message_chain:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

# Store the prompt or query from the user
query = st.chat_input('Ask questions about the product...')

if query is not None and query!= "":
    # When an user inputs a prompt, this is categorized as Human
    with st.chat_message("Human"):
        st.markdown(query)
        # Create a prompt that contains information of previous messages and context
        st.session_state.history.append(HumanMessage(prompt_to_llm(query, faiss_index)))
        st.session_state.message_chain.append(HumanMessage(query))
        # Send the prompt to a openAI chat to obtain a response
        response = chat.invoke(st.session_state.history)
        
    with st.chat_message("AI"):
        # Use the response as the AI message that is going to be displayed
        bot_response = response.content
        parser = StrOutputParser()
        text_output = parser.parse(bot_response)
        st.markdown(text_output)
    # Add the messages to the history so it can be useful for future references
    st.session_state.history.append(AIMessage(bot_response))
    st.session_state.message_chain.append(AIMessage(bot_response))
with st.sidebar:
    st.markdown('## Feel free to download our catalog and ask questions regarding the products')
    st.download_button(label='Catalog', data=PDFbyte, file_name='Bruno_Child_Offers.pdf')
