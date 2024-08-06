# Start from the PyTorch base image
FROM pytorch/pytorch:latest
LABEL authors="crawfordcollins"

# Set the working directory in the container

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    supervisor \
    wget \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    curl \
    vim \
    python3 \
    python3-pip \
    python3-venv \
    libseccomp2 \
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
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH


# Install cloudflared
RUN curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb \
    && dpkg -i cloudflared.deb \
    && rm cloudflared.deb

# Clean up
RUN apt-get clean && rm -rf /var/lib/apt/lists/*


# Copy the application files to the working directory
WORKDIR /app
COPY src/ /app/src
COPY api.py /app/api.py
COPY prompts.py /app/prompts.py
COPY agent_splitter.py /app/agent_splitter.py
COPY instagram_util.py /app/instagram_util.py
COPY search.py /app/search.py
COPY settings.py /app/settings.py
COPY embeddings-cache /app/embeddings-cache

# Install Python packages
RUN pip3 install --no-cache-dir fastapi chromadb \
    pydantic anthropic lxml bs4 \
    llama-index-vector-stores-chroma llama-index-llms-groq \
    llama-index-embeddings-huggingface slowapi

RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu125

# Copy the supervisord configuration file
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Start supervisord when the container starts
ENTRYPOINT ["/usr/bin/supervisord"]
CMD ["-c", "/etc/supervisor/conf.d/supervisord.conf"]