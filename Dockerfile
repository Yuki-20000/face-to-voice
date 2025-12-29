FROM python:3.10-slim

# System Dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    espeak-ng \
    libsndfile1 \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Working Directory
WORKDIR /app

# Copy requirements first to leverage Docker cache.
COPY requirements.txt .

# Install Python packages. 
RUN pip install --no-cache-dir -r requirements.txt

# Copy Application Code
COPY . .

# Hugging Face Specific Setup
RUN chmod -R 777 /app

# Expose Port
EXPOSE 7860

# Run Streamlit. 
CMD ["streamlit", "run", "src/app.py", "--server.port=7860", "--server.address=0.0.0.0"]