from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS,faiss
from langchain.embeddings import SentenceTransformerEmbeddings,HuggingFaceInstructEmbeddings,HuggingFaceHubEmbeddings,OpenAIEmbeddings,HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.chains import QAWithSourcesChain,RetrievalQA
from langchain.docstore.document import Document
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
from tqdm import tqdm
from openai import OpenAI
from PDF_Read1 import create_index1,uploaded_docs,load_model,new_model,prepare_data
from summary_functions import prepare_data_summarize,get_page_summary
import pandas as pd
import numpy as np
import time
import os
import openai
import faiss
import json
from dotenv import load_dotenv
load_dotenv()
os.environ.get("OPENAI_API_KEY")
OPEN_API_KEY=os.environ.get("OPENAI_API_KEY")

app=FastAPI()

client=OpenAI(api_key=OPEN_API_KEY)
sys_prompt= """You are a research analyst and you are expert in reading the text of the document and summarize it in to 300 words."""
user_prompt="""You are provided with the PDF document your task is to summarize the provided document in 300 words"""


@app.post('/pdf_summarizer')
async def pdf_summary(file:UploadFile):
    page_summaries={}
    file_location=f"./{file.filename}"
    with open(file_location, "wb") as file_object:
         file_object.write(await file.read())
    My_pdf, len_pdf=uploaded_docs(file_location)
    docs=prepare_data_summarize(My_pdf)
    data_dict={}
    for document in docs:
        data_dict[document.metadata['source']]=document.page_content
## Getting page wise summary
    for page,content in data_dict.items():
        page_summary=get_page_summary(content)
        page_summaries[page]=page_summary
    final_summary=get_page_summary(page_summaries)    
    
    

app=FastAPI()

class PDF_File(BaseModel):
    query: str

# This will upload the pdf and create a FAISS Vectorstore for the excerpts of the PDF
save_path=''
@app.post('/uploadPDF_indexing')
async def pdf_file(file:UploadFile):
   file_location=f"./{file.filename}"
   with open(file_location, "wb") as file_object:
        file_object.write(await file.read())
   global save_path
   My_pdf, len_pdf=uploaded_docs(file_location)
   path1=create_index1(My_pdf,"new_index_LLM/")
   save_path=path1
   return (f"The file is successfully loaded and index created")  

# This endpoint is to Query the uploaded PDF
@app.post("/query_PDF")
async def query_file(item:PDF_File):
    chatbot=load_model("./new_index_LLM")
    result=chatbot.run(item.query)
    return result

# This Endpoint is used to upload the json file 
save_path1=""  
@app.post("/upload_json")
async def upload_file(file:UploadFile):
    file_location=f"./{file.filename}"
    global save_path1
    save_path1=file_location
    with open(file_location, "wb") as file_object:
        file_object.write(await file.read())
    return("Your Json file is uploaded successfully")

# This Endpoint is used for querying the uploaded json
@app.post("/query_json")
async def query_json(item:PDF_File):
    with open ("./data.json",'r') as f:
        data = json.load(f)

    result=new_model(data,item.query)
    return result
   

    

    







