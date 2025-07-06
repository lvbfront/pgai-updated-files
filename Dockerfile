FROM timescale/pgai-vectorizer-worker:latest
RUN pip install httpx
# Copy the updated files into the image
COPY pgai/vectorizer/vectorizer.py /app/pgai/vectorizer/vectorizer.py
COPY pgai/vectorizer/embedders/__init__.py /app/pgai/vectorizer/embedders/__init__.py
COPY pgai/vectorizer/embedders/fastapi_hf_local.py /app/pgai/vectorizer/embedders/fastapi_hf_local.py
