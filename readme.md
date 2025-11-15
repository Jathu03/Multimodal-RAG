# Multimodal RAG with PDF, CLIP, and Gemini

This project demonstrates a complete, end-to-end Retrieval-Augmented Generation (RAG) system capable of understanding and answering questions about both the text and the images contained within PDF documents.

It uses the CLIP model to generate unified embeddings for text chunks and images, stores them in a FAISS vector store, and leverages a powerful Large Language Model (like Google's Gemini) to synthesize answers based on the retrieved multimodal context.

## Features

- **PDF Processing**: Extracts both text and images from PDF files.
- **Unified Multimodal Embeddings**: Uses OpenAI's CLIP model to create comparable vector embeddings for both text and images.
- **Vector Storage**: Builds a FAISS vector store for efficient similarity searches across text and image content.
- **Persistent Storage**: Saves the generated vector store and image data to disk, separating the one-time ingestion process from querying.
- **Multimodal Context**: Constructs a prompt for the LLM that includes both relevant text excerpts and the actual images for comprehensive understanding.
- **Modular and Scalable**: The code is organized into a clean project structure for easy maintenance and expansion.

## Project Structure
