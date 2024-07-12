# Use Debian as the base image
FROM 637423200832.dkr.ecr.us-east-1.amazonaws.com/crawfordc:crawford-base AS high_life_base

# Set the working directory in the container
WORKDIR /app

# Copy the application files to the working directory
COPY src/ /app/src
COPY api.py /app/api.py
COPY prompts.py /app/prompts.py
COPY search.py /app/search.py
COPY settings.py /app/settings.py
COPY chroma_data/ /app/chroma_data
COPY embeddings-cache /app/embeddings-cache
COPY .cloudflared/ /app/cloudflared

# Install necessary system packages including Python 3 and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install Uvicorn and Chroma
RUN pip3 install --no-cache-dir fastapi chromadb \
    pydantic anthropic lxml bs4 \
    llama-index-vector-stores-chroma llama-index-llms-groq \
    llama-index-embeddings-huggingface

# Copy the supervisord configuration file
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Find the path to supervisord
RUN which supervisord > /supervisord_path.txt

# RUNTIME STAGE
FROM high_life_base AS high_life_runtime
# Start supervisord when the container starts
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]