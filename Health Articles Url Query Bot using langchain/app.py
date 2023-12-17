#loading all the necessary libraries
import os
import time
import pickle 
import ssl
import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import WebBaseLoader
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()
# building our application
st.title("🩺📚 Health Articles Querying Bot")
st.sidebar.title('Health Articles Urls')

#Getting the urls from tha user
urls=[]
for i in range(4):
    url=st.sidebar.text_input(f"Url{i+1}")
    urls.append(url)

process_url_clicked=st.sidebar.button('Process URLs')
main_placeholder=st.empty()
llm=OpenAI(temperature=0.9,max_tokens=1000)
if process_url_clicked:
    #loading the data          
    loader = UnstructuredURLLoader(urls,ssl_verify=False)
    main_placeholder.text('Data Loading...Started...✅✅✅')
    data=loader.load()
    #Splitting the data into chunks
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    text_splitter=RecursiveCharacterTextSplitter(
        separators=['/n/n','/n','.',','],
        chunk_size=1000
    )
    docs=text_splitter.split_documents(data)
    embeddings=OpenAIEmbeddings()
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    vectorstore=FAISS.from_documents(docs,embeddings)
    time.sleep(2)
    
query=main_placeholder.text_input('Questions:')
if query:
        chain=RetrievalQAWithSourcesChain(llm=llm,retriever=vectorstore.as_retriever())
        result=chain({"question":query},return_only_outputs=True)
        st.header("Answer")
        st.write(result['answer'])
