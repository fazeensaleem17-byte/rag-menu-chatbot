"""
main.py
-------
FastAPI application for AskTheMenu. This file only handles the WEB layer:
receiving HTTP requests, validating input, calling into rag_utils.py for the
actual RAG logic, and returning HTTP responses. Keeping the RAG logic in a
separate module makes both files easier to read on their own.

Endpoints:
  POST /upload  - upload a menu PDF, index it into the vector database
  POST /query   - ask a question about the currently uploaded menu
"""

import os

from dotenv import load_dotenv

# Load variables from the .env file (e.g. GOOGLE_API_KEY) into the process's
# environment BEFORE we import rag_utils, since rag_utils reads the API key
# from os.environ at import time.
load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import rag_utils

app = FastAPI(title="AskTheMenu API")

# ---------------------------------------------------------------------------
# CORS (Cross-Origin Resource Sharing)
# ---------------------------------------------------------------------------
# Our React frontend (running on e.g. http://localhost:5173, the default
# Vite dev server port) is a different "origin" than our backend
# (http://localhost:8000). Browsers block cross-origin requests by default
# for security. This middleware tells the browser "it's fine for these
# frontend origins to call this API", which is what lets the React app
# actually reach our /upload and /query endpoints.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def check_api_key():
    """
    Fail loudly (in the server logs) at startup if the Gemini API key is
    missing, rather than letting every request fail with a confusing error
    later. This does not crash the server -- it just prints a warning.
    """
    if not os.environ.get("GOOGLE_API_KEY"):
        print(
            "WARNING: GOOGLE_API_KEY is not set. Copy .env.example to .env "
            "and add your Gemini API key, or /upload and /query will fail."
        )


class QueryRequest(BaseModel):
    question: str


@app.post("/upload")
async def upload_menu(file: UploadFile = File(...)):
    """
    Accepts a PDF file, extracts its text, splits it into chunks, embeds
    each chunk, and stores the embeddings in ChromaDB.

    Returns: {"filename": ..., "chunks_created": ...}
    """
    # --- Validate the file is actually a PDF ---
    # We check both the filename extension and the declared content-type,
    # since a browser/client could get either one wrong; checking both
    # catches more bad input.
    is_pdf_extension = file.filename is not None and file.filename.lower().endswith(".pdf")
    is_pdf_content_type = file.content_type == "application/pdf"

    if not (is_pdf_extension or is_pdf_content_type):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted. Please upload a .pdf file.",
        )

    if not os.environ.get("GOOGLE_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="Server is missing GOOGLE_API_KEY. Add it to your .env file and restart the server.",
        )

    # --- Read the uploaded file's bytes ---
    try:
        file_bytes = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded file.")

    if not file_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    # --- Stage 1: extract text from the PDF ---
    try:
        text = rag_utils.extract_text_from_pdf(file_bytes)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read this PDF (it may be corrupted or scanned as images only). Error: {e}",
        )

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail=(
                "No readable text was found in this PDF. It may be a scanned "
                "image without a text layer, which this app cannot process."
            ),
        )

    # --- Stage 1.5: collapse repeated/corrupted text before chunking ---
    # Two different PDF export quirks can duplicate text before it ever
    # reaches chunking, both of which hurt retrieval if left alone:
    #   1. Individual CHARACTERS drawn multiple times on top of each other
    #      (e.g. a title/price rendered with a font-shadow effect) --
    #      "MMMMiiiigggghhhhttttyyyy" instead of "Mighty". Fixed first, so
    #      the word-level check below sees real words, not garbled ones.
    #   2. Whole WORDS/PHRASES repeated in a row (e.g. an item's name
    #      printed 4 times as a layout artifact) -- "Zinger Burger Zinger
    #      Burger Zinger Burger Zinger Burger" instead of "Zinger Burger".
    #      Left alone, this padding can push an item's price into a
    #      different chunk than its name, breaking retrieval for it.
    # See collapse_repeated_characters() and collapse_repeated_phrases()
    # in rag_utils.py for the full explanation of each, including a real
    # limitation: neither can recover text that two separate, overlapping
    # PDF elements (e.g. a price badge drawn on top of a description)
    # scrambled together into one interleaved, unrecoverable jumble --
    # that's data loss in the source PDF itself, not something any text
    # cleanup step can fix.
    text = rag_utils.collapse_repeated_characters(text)
    text = rag_utils.collapse_repeated_phrases(text)

    # --- Stage 2: split into chunks ---
    chunks = rag_utils.chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Could not split this PDF's text into chunks.")

    # --- Stage 3: embed chunks and store them in ChromaDB ---
    try:
        chunks_created = rag_utils.embed_and_store_chunks(chunks, file.filename)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to generate embeddings or store them (check your GOOGLE_API_KEY and network connection). Error: {e}",
        )

    return {"filename": file.filename, "chunks_created": chunks_created}


@app.post("/query")
async def query_menu(request: QueryRequest):
    """
    Accepts a question, retrieves the most relevant menu chunks from
    ChromaDB, and asks Gemini to answer using only those chunks.

    Returns: {"answer": ...}
    """
    question = (request.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not os.environ.get("GOOGLE_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="Server is missing GOOGLE_API_KEY. Add it to your .env file and restart the server.",
        )

    # --- Make sure a document has actually been uploaded first ---
    try:
        if not rag_utils.has_documents():
            raise HTTPException(
                status_code=400,
                detail="No menu has been uploaded yet. Please upload a menu PDF first.",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not access the document database. Error: {e}")

    # --- Stage 4a: retrieve the most relevant chunks ---
    try:
        relevant_chunks = rag_utils.retrieve_relevant_chunks(question, top_k=5)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to search the menu for relevant sections. Error: {e}",
        )

    if not relevant_chunks:
        raise HTTPException(
            status_code=400,
            detail="No menu has been uploaded yet. Please upload a menu PDF first.",
        )

    # --- Stage 4b: generate the answer with Gemini ---
    try:
        answer = rag_utils.generate_answer(question, relevant_chunks)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to get an answer from Gemini (check your GOOGLE_API_KEY and network connection). Error: {e}",
        )

    return {"answer": answer}


@app.get("/")
def root():
    """Simple health check so you can confirm the server is running."""
    return {"status": "AskTheMenu API is running"}
