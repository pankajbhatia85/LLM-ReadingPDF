# LLM-ReadingPDF
![image](https://github.com/pankajbhatia85/LLM-ReadingPDF/assets/85344841/5d12482c-a40b-401c-831c-9936dc213bd0)
# Document Summarizer

This repository provides a tool to summarize PDFs or manually entered user text via a frontend interface. The backend is powered by a `FASTAPI` endpoint called `document_summarizer`. Users must provide their OpenAI API key and install the required dependencies.

## Features

- Summarizes text from PDFs or user input.
- Uses OpenAI's API for generating summaries.
- Fast and reliable backend powered by `FASTAPI`.
- Customizable through user-provided OpenAI API keys.

## Prerequisites

1. Python 3.10 or higher.
2. An OpenAI API key.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/pankajbhatia85/LLM-ReadingPDF.git
   cd LLM-ReadingPDF
## Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate # On Windows, use `venv\\Scripts\\activate`
## Install the required dependencies:
pip install -r requirements.txt
## Set up your OpenAI API key:
OPENAI_API_KEY=your-api-key-here
# Usage
Start the FASTAPI server: If your project uses if __name__ == "__main__" to run the uvicorn server, you can start it directly by running:
python app.py
Alternatively, use the following command if uvicorn is not included in the script:
uvicorn main:app --reload
## Access the application:
The FASTAPI server will be available at http://127.0.0.1:8000.
Use the /document_summarizer endpoint to provide a PDF or text input for summarization.
## Environment Variables
OPENAI_API_KEY: Your OpenAI API key. This key is required to use the summarization feature.
# Future Enhancements
Add support for interacting with multiple PDFs simultaneously.
Implement Streamlit as the frontend for an enhanced user experience.


