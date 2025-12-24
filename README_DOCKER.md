# Multimodal RAG with PDF, CLIP, and Gemini

This project demonstrates a complete, end-to-end Retrieval-Augmented Generation (RAG) system capable of understanding and answering questions about both the text and the images contained within PDF documents.

It uses the CLIP model to generate unified embeddings for text chunks and images, stores them in a FAISS vector store, and leverages a powerful Large Language Model (like Google's Gemini) to synthesize answers based on the retrieved multimodal context.

## Features

- **PDF Processing**: Extracts both text and images from PDF files.
- **Unified Multimodal Embeddings**: Uses OpenAI's CLIP model to create comparable vector embeddings for both text and images.
- **Vector Storage**: Builds a FAISS vector store for efficient similarity searches across text and image content.
- **Persistent Storage**: Saves the generated vector store and image data to disk, separating the one-time ingestion process from querying.
- **Multimodal Context**: Constructs a prompt for the LLM that includes both relevant text excerpts and the actual images for comprehensive understanding.
- **API & Docker Support**: Includes a FastAPI server and Docker configuration for easy deployment.
- **Modular and Scalable**: The code is organized into a clean project structure for easy maintenance and expansion.

## Project Structure

```text
multimodal_rag_project/
├── data/                       # Place PDF documents here
│   └── multimodal_sample.pdf
├── vector_store/               # Generated embeddings and index
│   ├── faiss_index/
│   └── image_data.pkl
├── src/                        # Source code
│   ├── __init__.py
│   ├── api.py                  # FastAPI application endpoints
│   ├── config.py
│   ├── data_processing.py
│   ├── vector_store_utils.py
│   └── rag_pipeline.py
├── ingest.py                   # Script to process PDFs
├── app.py                      # Interactive CLI application
├── Dockerfile                  # Docker build configuration
├── docker-compose.yml          # Docker Compose configuration
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (API Keys)
├── .gitignore
└── README.md
```

## Setup and Installation

Follow these steps to get the project up and running on your local machine.

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/multimodal_rag_project.git
cd multimodal_rag_project
```

### 2. Create a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.
```bash
# For Unix/macOS
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the root of the project directory and add your Google Gemini API key.
```ini
Gemini_key="YOUR_GEMINI_API_KEY_HERE"
```

### 5. Add Your Data
Place the PDF documents you want to process inside the `data/` directory.

---

## Usage

You can interact with this project via the Command Line Interface (CLI) or via the API/Docker. In all cases, you must first ingest the data.

### Step 1: Ingest Data
Run the ingestion script to process PDFs, generate embeddings, and create the vector store. This must be done once, or whenever you add new documents to `data/`.

```bash
python ingest.py
```
*Note: This process may take some time initially as it downloads the CLIP model from Hugging Face.*

---

### Option A: Run Interactive CLI
To run the simple interactive command-line application:

```bash
python app.py
```
This script loads the vector store and allows you to chat with your documents in the terminal.

---

### Option B: API & Docker
This repository provides a FastAPI server for querying the RAG pipeline and a Docker-based deployment.

#### 1. Running the API Locally
Make sure requirements are installed and `ingest.py` has been run.

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

#### 2. Running with Docker
You can build and run the application in an isolated container.

**Build the image:**
```bash
docker build -t multimodal-rag:latest .
```

**Run the container:**
```bash
docker run -p 8000:8000 -e Gemini_key="YOUR_API_KEY" multimodal-rag:latest
```

#### 3. Running with Docker Compose (Recommended)
This is the easiest method for local development.

```bash
export GEMINI_KEY="your-api-key"
docker-compose up --build
```

#### API Endpoints
Once the server is running (locally or via Docker), the following endpoints are available:

- `GET /health`: Basic health check.
- `POST /ingest`: Trigger the ingestion process to (re)create the vector store from the `data/` folder.
- `POST /query`: Query the system.
    - **Body:** `{"query": "your question", "top_k": 5, "include_images": false}`
```
