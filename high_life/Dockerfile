FROM python:3

# Set the working directory in the container
WORKDIR /app

RUN apt-get update && apt-get install -y supervisor

RUN curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
	| tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
	&& echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
	|  tee /etc/apt/sources.list.d/ngrok.list \
	&&  apt update \
	&&  apt install ngrok


# Install Uvicorn and Chroma
RUN pip3 install fastapi chromadb  \
    pydantic anthropic lxml bs4\
    llama-index-vector-stores-chroma llama-index-llms-groq \
    llama-index-embeddings-huggingface

ARG ANTHROPIC_API_KEY
ARG GROQ_API_KEY
ARG NGROK_TOKEN

ENV ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
ENV GROQ_API_KEY=${GROQ_API_KEY}

RUN ngrok config add-authtoken ${NGROK_TOKEN}

# Copy the application files to the working directory
COPY src/ /app/src
COPY api.py /app/api.py
COPY prompts.py /app/prompts.py
COPY search.py /app/search.py
COPY settings.py /app/settings.py
COPY chroma_data/ /app/chroma_data
COPY embeddings-cache /app/embeddings-cache
EXPOSE 8001


# Copy the supervisord configuration file
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose the necessary port for the FastAPI server
RUN which supervisord > /supervisord_path.txt
# Start supervisord when the container starts
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]