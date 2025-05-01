# AskNITAP- Retrieval-Augmented Generation (RAG)

## Project Overview
AskNITAP is a smart, AI-powered virtual assistant designed to answer queries related to the institute. Built using **Retrieval-Augmented Generation (RAG)**, it retrieves relevant context from the NIT Andhra Pradesh website and generates accurate, coherent responses using a language model. The chatbot runs locally on the user's system and can be integrated into the institute's website.

## Features
- **Retrieval-Augmented Generation (RAG):** Combines a powerful search-based context retrieval system with a large language model (LLM) for accurate and contextually aware responses.
- **Web Integration:** A simple web interface allows users to interact with the chatbot through a button integrated into a mock-up of the NIT Andhra Pradesh website.
- **Local Inference:** All models (LLM and vector database) run locally using Ollama for privacy and better performance.
- **Scalable and Modular Design:** The chatbot is designed to be modular, allowing for easy updates and additions to data and models.

## Technologies Used
- **LangChain**: For implementing the RAG system and managing vector embeddings.
- **Nomic**: For embedding text chunks.
- **Chroma**: Vector database for storing and querying vector embeddings.
- **Django**: Backend framework for the web interface.
- **HTML/CSS**: Frontend for the web app.
- **Ollama**: Local inference for LLMs (e.g., LLaMA 3.1).
- **Python**: General-purpose programming language for integrating all components.

## Installation

### Prerequisites
- Python 3.8 or higher
- Django
- LangChain, Nomic, and Chroma (for the RAG pipeline)
- Ollama (for local LLM inference)

### Steps to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/aaryan-cs/AskNIT-AP
   pip install -r requirements.txt
2. *Run server*
   '''bash
   cd rag_app
   python manage.py runserver
3. Go to http://127.0.0.1:8000/

