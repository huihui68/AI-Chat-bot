# AI PDF Chatbot with GPT-4 and ChromaDB

An interactive chatbot built with Streamlit, GPT-4 API, and ChromaDB, allowing users to upload PDF documents, extract content, and engage in contextual conversations. Includes user authentication with login and registration functionality.

## Features

- **Upload and Chat**: Upload PDFs and ask questions based on the document's content.
- **Advanced AI Models**: Powered by GPT-4 for intelligent and contextual responses.
- **Vector Database**: Efficient document retrieval using ChromaDB.
- **Authentication**: Secure login and registration for personalized usage.

## Requirements

Install dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```
# Setup

## Add Your OpenAI API Key

1. Create a `.env` file in the project root directory.
2. Add your OpenAI API key in the following format:

```plaintext
   OPENAI_API_KEY=your_openai_api_key
```
## Run Streamlit App 
1. Open your terminal and run the following command:
```bash
streamlit run app.py
```
2. Open the browser to navigate to http://localhost:8501 to use the chatbot.

