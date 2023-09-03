from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS,faiss
from langchain.embeddings import SentenceTransformerEmbeddings,HuggingFaceInstructEmbeddings,HuggingFaceHubEmbeddings,OpenAIEmbeddings,HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.chains import QAWithSourcesChain,RetrievalQA
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain import embeddings
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
import os
import openai
import faiss
import json
from constants import *
openai.api_key=OPEN_API_KEY
# We are using sentence transformer embeddings from HuggingFace
embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2", model_kwargs = {'device': 'cpu'})

# The function to read the pdf page to page and splitting the excerpts to chunks to create documents 
def prepare_data(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100,length_function=len,separators='\n')
    processed_data = []
    for text , value in tqdm(data):
        splits = text_splitter.split_text(text)
        processed_data.extend(
            [
                Document(
                    page_content= texts,
                    metadata={"source": value}
                ) for texts in splits
            ]
        )
    return processed_data

# This function will create the FAISS index for the chunks and save the Vectors locally
def create_index1(data, save_path):
    docs = prepare_data(data=data)
    docsearch = FAISS.from_documents(docs, embedding)             #This part is vectorstore using FAISS
    docsearch.save_local(save_path)
    return save_path

# This function is to query the user input within the Vectorstore and return the top result from the K=5 best docs
def load_model(output_path: str):
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(
            openai_api_key=OPEN_API_KEY,
            temperature=0.85, model_name="gpt-3.5-turbo", max_tokens=1024,
        ),
        chain_type="stuff",
        retriever=FAISS.load_local(output_path, embedding)      #We are loading the saved index, retriever= db.as_retriever()
            .as_retriever(search_type="similarity", search_kwargs={"k":5} )     #This is retriever
    )

# This function is to upload a PDF file locally with page numbers and excerpts
def uploaded_docs(filename):
    if filename.endswith(".pdf"):
    # Read PDF file into DataFrame object
        My_pdf=PdfReader(filename)
        page_tot=len(My_pdf.pages)
        extract_data=[]
        for page_num in range (page_tot):
          page=My_pdf.pages[page_num]
          content=page.extract_text()
          extract_data.append((str(content),str(page_num+1)))
    else:
        print("File type not supported")
    return extract_data,page_tot

# PROMPT to read an uploaded json file
def new_model(data1,query1):
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=[
           {"role": "system", "content": f"Query the JSON file:\n{data1}"},
           {"role": "user", "content": f"{query1}"},
      
    ], max_tokens=1000,
    temperature=0,
    )
    
    response_message = response["choices"][0]["message"]["content"]
    return response_message