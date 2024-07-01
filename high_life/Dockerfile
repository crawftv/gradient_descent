FROM python:3

# Set the working directory in the container
WORKDIR /app

RUN apt-get update && apt-get install -y supervisor
# Install Uvicorn and Chroma
RUN pip3 install fastapi chromadb  \
    bs4 pydantic anthropic lxml \
    llama-index-llms-ollama selenium \
    llama-index-embeddings-huggingface \
    llama-index-vector-stores-chroma


ENV ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
ENV GROQ_API_KEY=${GROQ_API_KEY}
ENV INSTAGRAM_USERNAME=${INSTAGRAM_USERNAME}
ENV INSTAGRAM_PASSWORD=${INSTAGRAM_PASSWORD}

# Copy the application files to the working directory
COPY . /app

# Copy the supervisord configuration file
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose the necessary port for the FastAPI server
EXPOSE 8001

# Start supervisord when the container starts
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]