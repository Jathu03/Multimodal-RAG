# Use a lightweight Python image
FROM python:3.10-slim

WORKDIR /app

# Install only what the frontend needs (lighter image)
RUN pip install --no-cache-dir streamlit requests

# Copy frontend code
COPY frontend/ frontend/

# Expose Streamlit port
EXPOSE 8501

# Command to run Streamlit
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]