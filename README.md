
Integrating Unsupported Models into pg_ai via FastAPI

Table of Contents
Prerequisites

Step 1: Prepare Your FastAPI Embedding Server

Step 2: Modify pg_ai Source Code

Step 3: Build a Custom pg_ai Docker Image

Step 4: Define the PostgreSQL  Functions

Step 5: Create the pg_ai Vectorizer

Step 6: Usage and Verification

Prerequisites
Before you begin, ensure you have:

A running PostgreSQL database with pg_ai installed.

docker-compose.yml

Your FastAPI embedding server running and accessible. This server should expose endpoints for generating embeddings (e.g., /embed for single texts and /batch_embed for multiple texts).

Docker and Docker Compose installed on your system.


Step 1: Prepare Your FastAPI Embedding Server
Your FastAPI server acts as the bridge between pg_ai and your chosen embedding model (e.g., paraphrase-MiniLM-L3-v2).

Key Requirements for Your FastAPI Server:

Endpoints: It must expose at least a /embed endpoint for single text embedding and preferably a /batch_embed endpoint for efficient batch processing.

Request Format: It should accept JSON bodies like {"text": "..."} or {"texts": ["...", "..."]}.

Response Format: It must return a JSON response containing the embedding(s) as a list of floats.

For /embed: {"embedding": [float, float, ...], "dimensions": int}

For /batch_embed: {"embeddings": [[float, ...], [float, ...]], "dimensions": int}

Install FastAPI Server Dependencies:

Before running your FastAPI server, ensure you install all necessary Python packages. This typically includes fastapi, uvicorn, and sentence-transformers (which will also pull in transformers and torch).

pip install fastapi uvicorn "sentence-transformers[torchensemble]" pydantic

Your FastAPI Server Code :

# main.py 

``` bash
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn
import logging
from typing import List
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global variable for the model
_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    logger.info("Loading SentenceTransformer model...")
    try:
        _model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise
    yield
    logger.info("Shutting down, cleaning up model...")
    _model = None

app = FastAPI(lifespan=lifespan)

class EmbeddingRequest(BaseModel):
    text: str

class BatchEmbeddingRequest(BaseModel):
    texts: List[str]

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "paraphrase-MiniLM-L3-v2"}

@app.post("/embed")
async def embed(request: EmbeddingRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty or whitespace")
    
    logger.info(f"Processing embedding request for text: {request.text[:50]}...")
    try:
        embedding = _model.encode(request.text, convert_to_numpy=True)
        logger.info(f"Embedding generated, dimensions: {len(embedding)}")
        return {
            "embedding": embedding.tolist(),
            "dimensions": len(embedding)
        }
    except Exception as e:
        logger.error(f"Embedding failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

@app.post("/batch_embed")
async def batch_embed(request: BatchEmbeddingRequest):
    if not request.texts:
        raise HTTPException(status_code=400, detail="texts list cannot be empty")
    
    logger.info(f"Processing batch embedding request with {len(request.texts)} texts")
    try:
        embeddings = _model.encode(request.texts, convert_to_numpy=True, batch_size=32)
        logger.info(f"Batch embedding generated, dimensions: {len(embeddings[0]) if embeddings.size > 0 else 0}")
        return {
            "embeddings": embeddings.tolist(),
            "dimensions": len(embeddings[0]) if embeddings.size > 0 else 0
        }
    except Exception as e:
        logger.error(f"Batch embedding failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch embedding failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

    #run uvicorn main:app --reload it takes some time in the beginning to load the model
   ``` 

run in the terminal 'server' is the name of the python file
python -m uvicorn server:app --reload


Step 2: Modify pg_ai Source Code

You need to tell pg_ai how to interact with your FastAPI server. This involves creating a new embedder definition and modifying the core vectorizer logic to use it.

the files structure
![image](https://github.com/user-attachments/assets/bcb179da-d5b3-4a7a-be79-31cbc9e71447)

Create pgai/vectorizer/embedders/fastapi_hf_local.py
This file defines the interface for your custom FastAPI embedder.

# pgai/vectorizer/embedders/fastapi_hf_local.py

```bash

from collections.abc import AsyncGenerator, Sequence
from typing import Literal

from pydantic import BaseModel
from typing_extensions import override

from ..embeddings import Embedder, EmbeddingResponse, EmbeddingVector, Usage, logger

class FastAPIHFLocal(BaseModel, Embedder):
    """
    Embedder that uses a FastAPI-based Hugging Face local server to embed documents into vector representations.

    Attributes:
        implementation (Literal["fastapi_hf_embed"]): The literal identifier for this implementation.
        url (str): The URL of the FastAPI server.
        dimensions (int): The dimensionality of the embeddings.
    """

    implementation: Literal["fastapi_hf_embed"]
    url: str
    dimensions: int

    @override
    async def embed(
        self, documents: list[str]
    ) -> AsyncGenerator[list[EmbeddingVector], None]:
        """
        This method is not used as the embedding logic is handled in vectorizer.py.
        """
        raise NotImplementedError("Embedding is handled in vectorizer.py for fastapi_hf_embed")

    @override
    def _max_chunks_per_batch(self) -> int:
        # Arbitrary default, can be tuned if needed
        return 2048

    @override
    async def setup(self):
        # No setup required for FastAPIHFLocal
        pass

    @override
    async def call_embed_api(self, documents: list[str]) -> EmbeddingResponse:
        # Not used, but required by the interface
        raise NotImplementedError("Embedding is handled in vectorizer.py for fastapi_hf_embed")

    async def _context_length(self) -> int | None:
        # Arbitrary value; can be adjusted if needed
        return None
```


Update pgai/vectorizer/embedders/__init__.py
This file makes your new embedder available for pg_ai to recognize.

# pgai/vectorizer/embedders/__init__.py

```bash
from .fastapi_hf_local import FastAPIHFLocal as FastAPIHFLocal # just Add this line
```


Modify pgai/vectorizer/vectorizer.py
This is the core logic where pg_ai decides which embedder to use. You need to add specific logic to directly call your FastAPI server when fastapi_hf_embed is selected.

```bash

# pgai/vectorizer/vectorizer.py (excerpt from _generate_embeddings method)
# Add this import at the top of the file if not already present
import httpx
import numpy as np # Ensure numpy is imported

# ... (rest of the Executor class and _generate_embeddings method) ...

async def _generate_embeddings(
    self, items: list[SourceRow]
) -> AsyncGenerator[
    tuple[list[EmbeddingRecord], list[tuple[SourceRow, LoadingError]]], None
]:
    # ... (existing code for processing items into documents) ...

    if loading_errors:
        yield [], loading_errors

    try:
        if self.vectorizer.config.embedding.implementation == "fastapi_hf_embed":
            fastapi_embedder_config = self.vectorizer.config.embedding
            fastapi_url = fastapi_embedder_config.url

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{fastapi_url}/batch_embed", # Use batch_embed for efficiency
                    json={"texts": documents},
                    timeout=60.0 # Adjust timeout as needed
                )
                response.raise_for_status() # Raise an exception for HTTP errors

                embeddings_data = response.json()
                returned_embeddings = embeddings_data.get("embeddings")

                if not isinstance(returned_embeddings, list):
                    raise ValueError("FastAPI response 'embeddings' is not a list")

                records_with_embeddings: list[EmbeddingRecord] = []
                for record_base, embedding_list in zip(
                    records_without_embeddings, returned_embeddings, strict=True
                ):
                    records_with_embeddings.append(record_base + [np.array(embedding_list)])
                yield records_with_embeddings, []
        else:
            # Existing logic for other embedders (OpenAI, Ollama, etc.)
            rwe_take = flexible_take(records_without_embeddings)
            async for embeddings_batch in self.vectorizer.config.embedding.embed(documents):
                records: list[EmbeddingRecord] = []
                for record, embedding in zip(
                    rwe_take(len(embeddings_batch)), embeddings_batch, strict=True
                ):
                    records.append(record + [np.array(embedding)])
                yield records, []

    except Exception as e:
        raise EmbeddingProviderError() from e

```

Step 3: Build a Custom pg_ai Docker Image

After modifying the pg_ai source code locally, you need to build a new Docker image for your vectorizer-worker service that includes these changes.

Install httpx for the vectorizer-worker:


# Dockerfile
FROM timescale/pgai-vectorizer-worker:latest

# Install httpx for the vectorizer worker to make HTTP requests
RUN pip install httpx

# Copy the updated files into the image
COPY pgai/vectorizer/vectorizer.py /app/pgai/vectorizer/vectorizer.py
COPY pgai/vectorizer/embedders/__init__.py /app/pgai/vectorizer/embedders/__init__.py
COPY pgai/vectorizer/embedders/fastapi_hf_local.py /app/pgai/vectorizer/embedders/fastapi_hf_local.py


# example for docker-compose.yml
name: pgai-test-env # A unique name for your test project

services:
  db-test: # Renamed database service
    image: timescale/timescaledb-ha:pg17
    environment:
      POSTGRES_PASSWORD: postgres
      # You can omit API keys here if your FastAPI server doesn't need them directly from DB
      # or if you prefer to manage them in the vectorizer-worker-test service.
    ports:
      - "5433:5432" # Changed host port to avoid conflict with main DB (5432)
    volumes:
      - data-test:/home/postgres/pgdata/data # New named volume for test data
    command: [ "-c", "ai.ollama_host=http://ollama:11434" ] # Keep if you use ollama

  vectorizer-worker-test: # Renamed vectorizer worker service
    image: my-test-vectorizer-worker:latest # change this
    environment:
      PGAI_VECTORIZER_WORKER_DB_URL: postgres://postgres:postgres@db-test:5432/postgres # Connects to the test DB
      # Include any other environment variables your worker needs, e.g., API keys
      OLLAMA_HOST: http://ollama:11434
      GEMINI_API_KEY: ''
      HUGGINGFACE_API_KEY: ''
      PATH: /home/pgaiuser/.local/bin:$PATH
    volumes:
      # If your credentials.json is critical and needs to be mounted, adjust path
      - /home/awb9/pgai/credentials.json:/app/credentials.json
    command: [ "--poll-interval", "5s", "--log-level", "DEBUG" ] # Keep your worker command

  ollama: # Keep ollama if your FastAPI or other services use it
    image: ollama/ollama
    deploy:
      resources:
        limits:
          memory: 6g

  pgadmin-test: # Renamed pgAdmin service
    image: dpage/pgadmin4
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@admin.com
      PGADMIN_DEFAULT_PASSWORD: admin
    ports:
      - "5051:80" # Changed host port to avoid conflict with main pgAdmin (5050)
    depends_on:
      - db-test # Depends on the test DB
      
volumes:
  data-test: # New named volume for the test database

  

Command to build the new image:


Navigate to the directory containing your Dockerfile and run:


ensure it is the same name as in that  docker-compose.yml file

  vectorizer-worker-test: # Renamed vectorizer worker service
    image: my-test-vectorizer-worker:latest # change this
    
``` bash
docker build -t my-test-vectorizer-worker:latest .
```

ensure the name of the  docker-compose.yml file in my case it is docker-compose.test.yml
```bash
docker-compose -f docker-compose.test.yml up -d
```

in the postgresql

```bash
CREATE EXTENSION IF NOT EXISTS ai CASCADE;
```
in the terminal

docker-compose -f docker-compose.test.yml run --rm --entrypoint "python -m pgai install -d <your db link>" vectorizer-worker-test
docker compose up -d







Step 5: Define the PostgreSQL ai.fastapi_hf_embed Function
This function acts as the SQL-side interface that pg_ai uses to know how to call your FastAPI embedder. It uses PL/Python3u to make the actual HTTP request.

Prerequisite: Enable the PL/Python3u extension in your PostgreSQL database if you haven't already:

CREATE EXTENSION IF NOT EXISTS plpython3u;

Install requests for PL/Python3u:

docker exec -it <db_container_name_or_id> bash
# Inside the container:
pip install requests
exit



PostgreSQL Function Definition:
``` bash
-- FUNCTION: ai.fastapi_hf_embed(url text, input_text text, api_key text, input_type text, verbose boolean)
-- This function is crucial for pg_ai to interact with your FastAPI server.

CREATE OR REPLACE FUNCTION ai.fastapi_hf_embed(
    url text, -- The URL of your FastAPI /embed endpoint (e.g., 'http://host.docker.internal:8000/embed')
    input_text text, -- The text to be embedded
    api_key text DEFAULT NULL::text, -- Optional: API key for your FastAPI server if it requires authentication
    input_type text DEFAULT NULL::text, -- Optional: Any specific input type if your FastAPI supports it
    "verbose" boolean DEFAULT false -- Optional: For verbose logging within the PL/Python function
)
RETURNS vector -- This indicates the function returns a pgvector type
LANGUAGE 'plpython3u' -- Specifies that the function body is written in Python 3
COST 100
IMMUTABLE PARALLEL SAFE -- IMMUTABLE is fine here as the function itself doesn't change,
                        -- but the *result* of the HTTP call can vary, so VOLATILE might be more accurate.
                        -- However, for pg_ai's internal caching, IMMUTABLE might be desired.
AS $BODY$
import requests
import json
import sys
import sysconfig
from pathlib import Path

# --- pg_ai internal setup (keep as is if it's part of your pg_ai distribution) ---
if "ai.version" not in GD:
    r = plpy.execute("select coalesce(pg_catalog.current_setting('ai.python_lib_dir', true), '/usr/local/lib/pgai') as python_lib_dir")
    python_lib_dir = r[0]["python_lib_dir"]
    if "purelib" in sysconfig.get_path_names() and sysconfig.get_path("purelib") in sys.path:
        sys.path.remove(sysconfig.get_path("purelib"))
    python_lib_dir = Path(python_lib_dir).joinpath("0.10.1") # Ensure this version matches your pg_ai installation
    import site
    site.addsitedir(str(python_lib_dir))
    from ai import __version__ as ai_version
    assert("0.10.1" == ai_version) # Assert the version matches
    GD["ai.version"] = "0.10.1"
else:
    if GD["ai.version"] != "0.10.1":
        plpy.fatal("the pgai extension version has changed. start a new session")
# --- End pg_ai internal setup ---

import ai.utils # Assuming ai.utils is available in your pg_ai environment

if input_text is None or not isinstance(input_text, str):
    plpy.error("Input text must be a non-null string")

stripped_text = input_text.strip()
if not stripped_text:
    plpy.error("Input text cannot be empty after stripping")

data = {"text": stripped_text} # Payload for your FastAPI /embed endpoint
headers = {}
if api_key:
    headers["Authorization"] = f"Bearer {api_key}"

with ai.utils.VerboseRequestTrace(plpy, "fastapi_hf_embed", verbose):
    try:
        # Make the HTTP POST request to your FastAPI server
        response = requests.post(url, json=data, headers=headers, timeout=10)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        result = response.json()
    except requests.exceptions.RequestException as e:
        plpy.error(f"Request failed: {str(e)}") # Log and raise PostgreSQL error

embedding = result.get("embedding") # Extract the 'embedding' list from FastAPI response
if not isinstance(embedding, list) or not all(isinstance(x, (float, int)) for x in embedding):
    plpy.error("Invalid embedding format: expected a list of floats")

return embedding # PL/Python automatically converts Python lists/NumPy arrays to PostgreSQL vector type
$BODY$;

ALTER FUNCTION ai.fastapi_hf_embed(text, text, text, text, boolean)
    OWNER TO postgres;

```
edit validate_embeding function
``` bash
-- FUNCTION: ai._validate_embedding(jsonb)

-- DROP FUNCTION IF EXISTS ai._validate_embedding(jsonb);

CREATE OR REPLACE FUNCTION ai._validate_embedding(
	config jsonb)
    RETURNS void
    LANGUAGE 'plpgsql'
    COST 100
    IMMUTABLE PARALLEL UNSAFE
    SET search_path=pg_catalog, pg_temp
AS $BODY$
declare
    _config_type pg_catalog.text;
    _implementation pg_catalog.text;
begin
    if pg_catalog.jsonb_typeof(config) operator(pg_catalog.!=) 'object' then
        raise exception 'embedding config is not a jsonb object';
    end if;

    _config_type = config operator(pg_catalog.->>) 'config_type';
    if _config_type is null or _config_type not in ('embedding', 'custom_local') then
        raise exception 'invalid config_type for embedding config (debug: config_type = %)', _config_type;
    end if;

    _implementation = config operator(pg_catalog.->>) 'implementation';
    if _config_type = 'embedding' then
        case _implementation
            when 'openai' then
                -- ok
            when 'ollama' then
                -- ok
            when 'voyageai' then
                -- ok
            when 'litellm' then
                -- ok
            else
                if _implementation is null then
                    raise exception 'embedding implementation not specified';
                else
                    raise exception 'invalid embedding implementation: "%"', _implementation;
                end if;
        end case;
    elsif _config_type = 'custom_local' then
        if _implementation not in ('huggingface_embed_try2', 'fastapi_hf_embed') then
            raise exception 'invalid custom_local implementation: "%"', _implementation;
        end if;
        if config operator(pg_catalog.->>) 'implementation' is null or config operator(pg_catalog.->>) 'dimensions' is null then
            raise exception 'missing implementation or dimensions for custom_local embedding';
        end if;
    end if;
end
$BODY$;

ALTER FUNCTION ai._validate_embedding(jsonb)
    OWNER TO postgres;

GRANT EXECUTE ON FUNCTION ai._validate_embedding(jsonb) TO PUBLIC;

GRANT EXECUTE ON FUNCTION ai._validate_embedding(jsonb) TO postgres;

```

# postgresql function
``` bash
-- Drop the old SQL helper function if it exists
DROP FUNCTION IF EXISTS ai.embedding_fastapi(text, integer, text, jsonb, text);

-- Create or replace the SQL helper function
-- This function is called by ai.create_vectorizer to generate the JSONB config.
CREATE OR REPLACE FUNCTION ai.embedding_fastapi(
    model_name text,     -- e.g., 'paraphrase-MiniLM-L3-v2'
    dimensions integer,  -- e.g., 384
    fastapi_base_url text DEFAULT NULL::text, -- The base URL of your FastAPI server (e.g., 'http://host.docker.internal:8000')
    options jsonb DEFAULT NULL::jsonb,
    keep_alive text DEFAULT NULL::text
)
RETURNS jsonb
LANGUAGE 'sql'
COST 100
IMMUTABLE PARALLEL UNSAFE
SET search_path=pg_catalog, pg_temp
AS $BODY$
    SELECT json_strip_nulls(json_build_object(
        'implementation', 'fastapi_hf_embed', -- This tells pg_ai to use the 'fastapi_hf_embed' Python implementation
        'config_type', 'custom_local',
        'model', model_name,
        'dimensions', dimensions,
        'url', COALESCE(fastapi_base_url, 'http://host.docker.internal:8000'), -- This URL will be passed to the PL/Python function
        'options', options,
        'keep_alive', keep_alive
    ))
$BODY$;

ALTER FUNCTION ai.embedding_fastapi(text, integer, text, jsonb, text)
    OWNER TO postgres;


```

Execute this SQL statement in your PostgreSQL client.

Step 6: Create the pg_ai Vectorizer

``` bash
-- Create the new vectorizer
SELECT ai.create_vectorizer(
    source => 'test_documents'::regclass,
    name => 'test_documents_vectorizer'::text,
    destination => ai.destination_column('embedding_vector'),
    loading => ai.loading_column('text_content'),
    parsing => ai.parsing_auto(),
    embedding => ai.embedding_fastapi(
        'paraphrase-MiniLM-L3-v2',                -- model_name (text)
        384,                                      -- dimensions (integer)
        'http://host.docker.internal:8000'        -- fastapi_base_url (text)
        -- options and keep_alive are optional and can be omitted or set to NULL
    ),
    chunking => ai.chunking_none(),
    indexing => ai.indexing_none(),
    formatting => ai.formatting_python_template(),
    processing => ai.processing_default(),
    queue_schema => NULL::name,
    queue_table => NULL::name,
    grant_to => ai.grant_to(),
    enqueue_existing => true,
    if_not_exists => false
);
```

Step 7: Usage and Verification
INSERT INTO test_documents (text_content) VALUES ('This is the first test document.');
INSERT INTO test_documents (text_content) VALUES ('Another document for testing embeddings.');
INSERT INTO test_documents (text_content) VALUES ('The quick brown fox jumps over the lazy dog.');
INSERT INTO test_documents (text_content) VALUES ('A lazy cat slept soundly on the mat.');


Verify Embeddings:

``` bash
WITH query_embedding_vector AS (
    SELECT ai.fastapi_hf_embed(
        'http://host.docker.internal:8000/embed',
        'Animals that like to sleep.', -- Your search query
        NULL::text,
        NULL::text,
        false
    ) AS embedding_vector
)
SELECT
    td.id,
    td.text_content,
    (1 - (td.embedding_vector <=> q.embedding_vector)) AS cosine_similarity_score
FROM
    test_documents td,
    query_embedding_vector q
ORDER BY
    td.embedding_vector <=> q.embedding_vector
LIMIT 3;
```
Conclusion
By following these steps, you can successfully integrate any embedding model accessible via a FastAPI server into your pg_ai setup. This provides a flexible and scalable solution for managing and searching vector embeddings within your PostgreSQL database.
