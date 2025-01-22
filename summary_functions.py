from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS,faiss
from langchain.embeddings import SentenceTransformerEmbeddings,HuggingFaceInstructEmbeddings,HuggingFaceHubEmbeddings,OpenAIEmbeddings,HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.chains import QAWithSourcesChain,RetrievalQA
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain import embeddings
from tqdm import tqdm
from openai import OpenAI
from collections import defaultdict
import pandas as pd
import numpy as np
import time
import os
import openai
import faiss
import json
from constants import *
from dotenv import load_dotenv
load_dotenv()
os.environ.get("OPENAI_API_KEY")
openai.api_key=OPEN_API_KEY

client=OpenAI(api_key=OPEN_API_KEY)

# We are using sentence transformer embeddings from HuggingFace
embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2", model_kwargs = {'device':'cpu'})

# The function to read the pdf page to page and splitting the excerpts to chunks to create documents 
def prepare_data_summarize(data):
    combined_data = defaultdict(str)
    for text, value in tqdm(data):
        combined_data[value] += text + "\n" 
#    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100,length_function=len)
    processed_data = []
    for source, combined_text in combined_data.items(): # type: ignore
#        splits = text_splitter.split_text(combined_text)
        processed_data.extend(
            [
                Document(
                    page_content=combined_text,
                    metadata={"source": source}
                ) 
            ]
        )
    
    return processed_data

def get_page_summary(content):
    sys_prompt= """You are a research analyst and you are expert in reading the text of the document and summarize it in to 300 words."""
    
    if isinstance(content,str):
        user_prompt=f"""You are provided with the PDF text:
     content: {content} 
     Your task is to summarize the provided document in 300 words"""
    else:
        user_prompt=f"""You are provided with the dictionary with key as the page numbers and values are the summaries of the respective page:
     content: {content} 
     Your task is to combine all the summaries and provide a collated final summary of the document in 300 words"""
    response=client.chat.completions.create(model="gpt-4o",messages=[
        {
        "role":"system", "content":sys_prompt
        },
        {
        "role":"user",
        "content":user_prompt
    }
    ]
    )
    return response.choices[0].message.content

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