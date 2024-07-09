# Use Debian as the base image
FROM debian:bullseye-slim

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

# Copy the cloudflared startup script
COPY start_cloudflared.sh /app/start_cloudflared.sh
RUN chmod +x /app/start_cloudflared.sh

# Install necessary system packages including Python 3 and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    supervisor \
    wget \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install a newer version of SQLite
RUN wget https://www.sqlite.org/2023/sqlite-autoconf-3420000.tar.gz \
    && tar xvfz sqlite-autoconf-3420000.tar.gz \
    && cd sqlite-autoconf-3420000 \
    && ./configure --prefix=/usr/local \
    && make \
    && make install \
    && cd .. \
    && rm -rf sqlite-autoconf-3420000 sqlite-autoconf-3420000.tar.gz

# Update library path
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Install cloudflared
RUN curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb \
    && dpkg -i cloudflared.deb \
    && rm cloudflared.deb

# Install Uvicorn and Chroma
RUN pip3 install --no-cache-dir fastapi chromadb \
    pydantic anthropic lxml bs4 \
    llama-index-vector-stores-chroma llama-index-llms-groq \
    llama-index-embeddings-huggingface

# Set environment variables
ARG ANTHROPIC_API_KEY
ARG GROQ_API_KEY
ARG CLOUDFLARE_TOKEN

ENV ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
ENV GROQ_API_KEY=${GROQ_API_KEY}
ENV CLOUDFLARE_TOKEN=${CLOUDFLARE_TOKEN}


# Expose the necessary port for the FastAPI server
EXPOSE 8001

# Copy the supervisord configuration file
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Find the path to supervisord
RUN which supervisord > /supervisord_path.txt

# Install cloudflared service
RUN cloudflared service install $CLOUDFLARE_TOKEN

# Start supervisord when the container starts
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]