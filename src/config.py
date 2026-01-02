# --- Standard Library Imports ---
import os

# --- Third-Party Library Imports ---
from transformers import CLIPProcessor, CLIPModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("Gemini_key")

# --- Model & Processor Initialization ---
print("Initializing models and processors...")
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
GEMINI_MODEL_ID = "gemini-2.5-flash"

# Initialize once to save resources
CLIP_MODEL = CLIPModel.from_pretrained(CLIP_MODEL_ID)
CLIP_PROCESSOR = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
CLIP_MODEL.eval()  # Set the model to evaluation mode

LLM = ChatGoogleGenerativeAI(model=GEMINI_MODEL_ID)

# --- Text Splitter Configuration ---
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# --- File Path Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCUMENTS_PATH = os.path.join(BASE_DIR, "data")
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vector_store", "faiss_index")
IMAGE_STORE_PATH = os.path.join(BASE_DIR, "vector_store", "image_data.pkl")


if __name__ == "__main__":
    print("Config summary:")
    print("CLIP_MODEL_ID:", CLIP_MODEL_ID)
    print("GEMINI_MODEL_ID:", GEMINI_MODEL_ID)
    print("DOCUMENTS_PATH:", DOCUMENTS_PATH)
    print("VECTOR_STORE_PATH:", VECTOR_STORE_PATH)
    print("IMAGE_STORE_PATH:", IMAGE_STORE_PATH)
    try:
        _ = CLIP_MODEL
        print("CLIP_MODEL instance is loaded.")
    except Exception as e:
        print("CLIP_MODEL not available:", e)