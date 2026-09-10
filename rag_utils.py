"""
rag_utils.py
------------
This file contains all the "RAG" (Retrieval Augmented Generation) logic for
AskTheMenu, separated out from the FastAPI routes in main.py.

WHAT IS RAG, IN ONE PARAGRAPH?
Large Language Models (LLMs) like Gemini are very good at generating fluent
text, but they only know what they were trained on -- they have never seen
your restaurant's menu PDF. RAG fixes this by doing two things at answer
time: (1) RETRIEVE the small handful of pieces of your document that are
most relevant to the user's question, then (2) AUGMENT the LLM's prompt by
pasting those pieces in as "context" before asking the question. The LLM
then GENERATES an answer using only that context, instead of guessing from
its training data. This is why RAG chatbots can answer questions about
documents they were never trained on, and why they can say "I don't know"
when the document doesn't contain the answer.

The pipeline in this file has four stages, in the order you'll use them:
  1. Extract text from the uploaded PDF        (extract_text_from_pdf)
  2. Split that text into small "chunks"        (chunk_text)
  3. Turn chunks into embedding vectors and
     store them in a vector database            (embed_and_store_chunks)
  4. Given a question, retrieve the closest
     chunks and ask Gemini to answer from them  (retrieve_relevant_chunks,
                                                   generate_answer)
"""

import os
import re
import chromadb
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pdfplumber

# ---------------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------------

# Configure the Gemini client with our API key. This key is read from the
# environment (i.e. from the .env file) by main.py using python-dotenv, and
# by the time this module is used, os.environ["GOOGLE_API_KEY"] is already
# set. We never hardcode the key in source code.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# The Gemini models used for this project.
# - EMBEDDING_MODEL turns text into a list of numbers (a "vector") that
#   captures its meaning. Similar meanings -> similar vectors.
# - GENERATION_MODEL is the chat model that actually writes the final answer.
#
# NOTE: the original spec called for "models/text-embedding-004" and
# "gemini-1.5-flash". Both have since been retired by Google (confirmed via
# ListModels against this project's API key -- neither appears in the
# available model list anymore), so they were swapped for their current
# equivalents: gemini-embedding-001 (today's Gemini embedding model) and,
# originally, gemini-flash-latest for generation.
#
# UPDATE: gemini-flash-latest turned out to be the wrong choice. It's an
# alias, not a pinned model -- it silently resolved to gemini-3.8-flash
# (Google's newest, priciest Flash model at the time), which has a very
# small free-tier daily quota (we hit a 429 "quota exceeded... limit: 20"
# after normal testing).
#
# Tried pinning to gemini-2.5-flash next, but that model has since been
# cut off from new users/keys entirely (confirmed via a direct API call --
# it 404s with "no longer available to new users", even though it still
# shows up in ListModels, so ListModels alone isn't proof a model is
# actually usable).
#
# Settled on gemini-3.1-flash-lite: a specific PINNED version (won't drift
# like a "-latest" alias can), on the "lite" tier rather than full "flash"
# (lite tiers are priced/quota'd for higher-volume, lower-cost use, so
# they typically carry a more generous free-tier daily request limit than
# the full flash tier that just got rate-limited), and confirmed working
# with grounded, correctly-refusing answers in testing. Google no longer
# publishes exact free-tier RPD/RPM numbers in its docs (confirmed by
# checking directly) -- they're account-specific now, visible at
# https://aistudio.google.com/rate-limit -- so if you hit a 429 again,
# check that page for this key's actual current limit on this model.
EMBEDDING_MODEL = "models/gemini-embedding-001"
GENERATION_MODEL = "gemini-3.1-flash-lite"

# ChromaDB is our vector database: a database specialised for storing
# embedding vectors and quickly finding "which stored vectors are closest to
# this new vector?". We use PersistentClient so the data is saved to disk
# (in the ./chroma_db folder) and survives server restarts, instead of
# living only in memory.
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "menu_chunks"

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)


def get_or_create_collection():
    """
    A Chroma "collection" is like a table in a normal database -- it's where
    we store our chunk vectors plus their original text and metadata.
    We fetch-or-create it here so callers don't have to worry about whether
    it already exists.
    """
    return chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def reset_collection():
    """
    Deletes and recreates the collection.

    WHY: This project is designed around ONE menu being active at a time
    ("answers based only on that menu"). If we kept adding every uploaded
    PDF's chunks into the same collection forever, a later question could
    accidentally retrieve chunks from an old, unrelated menu and produce a
    wrong or confusing answer. So every time a new PDF is uploaded, we wipe
    the old chunks first and start fresh with just the new document.
    """
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        # Collection may not exist yet on the very first upload -- that's fine.
        pass
    return chroma_client.get_or_create_collection(name=COLLECTION_NAME)


# ---------------------------------------------------------------------------
# STAGE 1: EXTRACT TEXT FROM THE PDF
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Uses pdfplumber to pull raw text out of every page of the PDF and joins
    it into one big string.

    WHY pdfplumber specifically: menu PDFs are often laid out with columns,
    tables and prices lined up visually. pdfplumber is good at reading text
    in a sensible left-to-right, top-to-bottom order (better than just
    dumping characters in whatever order they appear in the PDF's internal
    structure), which matters a lot for getting "Chicken Biryani ... Rs 450"
    to come out as readable text instead of a jumble.
    """
    import io

    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

    return "\n".join(text_parts)


def collapse_repeated_characters(text: str, factor: int = 4) -> str:
    """
    Collapses runs of 4+ identical consecutive CHARACTERS down to their
    true repeat count -- e.g. "ZZZZiiiinnnnggggeeeerrrr" becomes "Zinger",
    and "RRRRssss 9999111100000000" becomes "Rs 9100".

    WHY THIS IS NEEDED (a second, different corruption pattern from
    collapse_repeated_phrases above): some menu PDFs aren't just laid out
    with repeated *words* -- they're exported from a styled web page (e.g.
    a delivery-app "print to PDF") where certain elements (headers, item
    titles, prices) use a font/shadow effect that draws each individual
    CHARACTER glyph multiple times almost on top of itself. pdfplumber
    faithfully extracts every one of those overlapping glyphs, so what
    should read "Mighty Zinger" comes out as
    "MMMMiiiigggghhhhttttyyyy ZZZZiiiinnnnggggeeeerrrr". Left alone, this
    makes item names and prices unrecognizable as text (and unrecognizable
    to embedding search), not just longer.

    We only treat runs of `factor` (4) or more identical characters as
    this kind of corruption -- ordinary English words essentially never
    have 4 identical letters in a row, so normal text (including genuine
    doubled letters, like the two D's in "ADD") is left untouched. We
    recover the true repeat count by dividing the run length by the
    corruption factor (rather than always collapsing to a single
    character), so a genuinely doubled letter under this same corruption
    -- "ADD" -> "AAAA" + "DDDDDDDD" (8 D's, since each of the 2 real D's
    was individually quadrupled) -- comes back as "ADD", not "AD". The
    same logic matters for prices with a repeated digit, e.g. "900".

    LIMITATION: this fixes characters duplicated *on top of themselves*.
    It cannot fix a different, more severe problem also seen in some PDFs
    (see main.py's upload handler comment): two entirely separate pieces
    of text -- e.g. a price badge overlapping a menu item's description --
    landing at overlapping positions on the page, which pdfplumber then
    extracts as a genuinely INTERLEAVED jumble of two different strings'
    characters. No text-only cleanup can safely undo that, because the
    original two strings can no longer be told apart character by
    character. That's a data-loss problem in the source PDF itself, not
    something chunking, retrieval, or preprocessing can fix.
    """
    def repl(match):
        run = match.group(0)
        char = match.group(1)
        true_count = max(1, len(run) // factor)
        return char * true_count

    return re.sub(r"(.)\1{" + str(factor - 1) + r",}", repl, text)


def collapse_repeated_phrases(text: str, max_phrase_words: int = 6) -> str:
    """
    Collapses immediately-repeated words/phrases down to a single occurrence,
    line by line -- e.g. "Zinger Burger Zinger Burger Zinger Burger Zinger
    Burger" becomes "Zinger Burger".

    WHY THIS IS NEEDED:
    Some menu PDFs render an item's name multiple times in a row on the same
    line (often a side effect of decorative/stylised layouts, where the
    name is drawn as a repeating background pattern behind the price).
    pdfplumber faithfully extracts all of that repeated text. Left alone,
    this padding pushes the item's price line further away from its name --
    far enough, in a long menu, that RecursiveCharacterTextSplitter's 500
    character budget can end up splitting the name into one chunk and the
    price into the next. Once that happens, no single retrieved chunk
    contains BOTH "Zinger Burger" and its price, and Gemini (correctly,
    given what it was shown) says it can't find a standalone price for it.
    Collapsing the repeated text back down to one occurrence removes that
    artificial padding, so the name and price are far more likely to land
    in the same chunk in the first place -- fixing the problem at its
    source instead of just working around it at query time.

    This only collapses text that repeats immediately next to itself (e.g.
    "X X X X"), so it won't touch normal menu text like "Chicken Burger
    Combo" or "Rs 700 Rs 750" (different numbers), which never take this
    repeated-phrase shape.
    """
    collapsed_lines = []
    for line in text.split("\n"):
        words = line.split()
        result = []
        i = 0
        while i < len(words):
            matched = False
            # Look for the SMALLEST repeating unit first (e.g. a 2-word
            # phrase repeating 4x), so "A B A B A B A B" collapses all the
            # way down to "A B" in one pass, not just halfway to "A B A B".
            max_len = min(max_phrase_words, (len(words) - i) // 2)
            for phrase_len in range(1, max_len + 1):
                phrase = words[i : i + phrase_len]
                next_phrase = words[i + phrase_len : i + 2 * phrase_len]
                if phrase == next_phrase:
                    # Found a repeat -- keep consuming as long as it repeats.
                    repeat_count = 2
                    while (
                        words[i + repeat_count * phrase_len : i + (repeat_count + 1) * phrase_len]
                        == phrase
                    ):
                        repeat_count += 1
                    result.extend(phrase)
                    i += repeat_count * phrase_len
                    matched = True
                    break
            if not matched:
                result.append(words[i])
                i += 1
        collapsed_lines.append(" ".join(result))
    return "\n".join(collapsed_lines)


# ---------------------------------------------------------------------------
# STAGE 2: SPLIT TEXT INTO CHUNKS
# ---------------------------------------------------------------------------

def chunk_text(text: str) -> list[str]:
    """
    Splits the full menu text into smaller overlapping "chunks" using
    LangChain's RecursiveCharacterTextSplitter.

    WHY WE CHUNK AT ALL:
    1. Embedding models and LLM context windows work better with small,
       focused pieces of text rather than one giant blob -- a single vector
       for an entire 5-page menu would blur together "starters", "desserts"
       and "drinks" into one vague meaning, making retrieval inaccurate.
    2. When we later retrieve "the most relevant chunks" for a question, we
       want each chunk to be small enough that it's ABOUT one specific
       thing (e.g. a couple of menu items), so the pieces we hand to Gemini
       are actually relevant and not full of irrelevant noise.

    WHY "RECURSIVE CHARACTER" SPLITTING SPECIFICALLY:
    RecursiveCharacterTextSplitter tries to split on natural boundaries
    first -- paragraph breaks ("\n\n"), then line breaks ("\n"), then
    spaces -- only falling back to splitting mid-word if it has to. This
    keeps related lines (like a dish name and its price/description) together
    in the same chunk, instead of chopping text at an arbitrary character
    count.

    WHY OVERLAP (chunk_overlap):
    Menu items near a chunk boundary could otherwise get cut in half, e.g.
    the chunk ends right after "Chicken Biryani" and the price "Rs 450"
    starts the next chunk. A bit of overlap means that boundary text appears
    in BOTH chunks, so nothing important gets lost between them.

    A NOTE ON THE LIMITS OF OVERLAP: overlap only helps when a name and its
    price are separated by a SMALL gap. It does not reliably reunite them
    when something (e.g. a long run of repeated/padding text -- see
    collapse_repeated_phrases() above) pushes a large gap between them;
    tested up to chunk_overlap=250 (half the chunk size, already too high
    for normal use -- it would make ChromaDB store heavily duplicated
    chunks) on a reproduction of exactly that scenario and the name and
    price still landed in different chunks. That's why this app also
    collapses repeated text before chunking, rather than relying on
    overlap alone.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # roughly how many characters per chunk
        chunk_overlap=100,    # characters shared between consecutive chunks
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    # Drop any empty/whitespace-only chunks (can happen with unusual PDFs).
    return [c.strip() for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# STAGE 3: EMBED CHUNKS AND STORE THEM IN CHROMADB
# ---------------------------------------------------------------------------

def embed_and_store_chunks(chunks: list[str], filename: str) -> int:
    """
    Turns each text chunk into an embedding vector via the Gemini embedding
    API, then stores (chunk text + its vector) in ChromaDB.

    WHAT IS AN EMBEDDING?
    An embedding is a list of a few hundred numbers (a "vector") that
    represents the MEANING of a piece of text, produced by a neural network
    trained for exactly this purpose. Text with similar meaning ends up with
    vectors that point in a similar "direction" in that numerical space --
    even if the two pieces of text don't share many of the same words. E.g.
    "veggie starters" and "vegetarian appetizers" would land close together.
    This is what lets us search by MEANING instead of by exact keyword match.

    task_type="retrieval_document": we tell the embedding model that this
    text is a *document being stored for later retrieval* (as opposed to a
    *search query*). Gemini's embedding model actually produces slightly
    different (better-suited) vectors depending on which side of the search
    you're embedding -- more on this in retrieve_relevant_chunks() below.

    Returns the number of chunks successfully stored.
    """
    if not chunks:
        return 0

    collection = reset_collection()

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=chunk,
            task_type="retrieval_document",
        )
        embeddings.append(result["embedding"])
        documents.append(chunk)
        ids.append(f"{filename}-chunk-{i}")
        metadatas.append({"filename": filename, "chunk_index": i})

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    return len(chunks)


# ---------------------------------------------------------------------------
# STAGE 4a: RETRIEVE THE MOST RELEVANT CHUNKS FOR A QUESTION
# ---------------------------------------------------------------------------

def has_documents() -> bool:
    """Returns True if at least one chunk has been uploaded and stored."""
    collection = get_or_create_collection()
    return collection.count() > 0


def retrieve_relevant_chunks(question: str, top_k: int = 5) -> list[str]:
    """
    Embeds the user's question, then asks ChromaDB for the top_k stored
    chunks whose vectors are closest to the question's vector.

    task_type="retrieval_query": this time we tell Gemini's embedding model
    that this text is a *search query*, not a document. Using the matching
    task_type on both sides (documents stored with "retrieval_document",
    questions embedded with "retrieval_query") is a Gemini-specific tuning
    that measurably improves search relevance versus embedding everything
    the same way.

    WHAT IS COSINE SIMILARITY?
    Once we have the question's vector, we need a way to measure "how close"
    it is to each stored chunk's vector. Cosine similarity measures the
    ANGLE between two vectors rather than their raw distance: it takes the
    dot product of the two vectors and divides by the product of their
    lengths. A result of 1 means the vectors point in exactly the same
    direction (very similar meaning), 0 means unrelated, -1 means opposite.
    This is the standard similarity metric for text embeddings because it
    ignores vector magnitude (which can vary with text length) and focuses
    purely on "do these two pieces of text mean similar things". ChromaDB's
    default collection is configured to use cosine similarity under the
    hood, which is why we don't have to compute it by hand -- collection
    .query() does it for us and returns the closest matches, ranked best
    first.
    """
    collection = get_or_create_collection()

    if collection.count() == 0:
        return []

    query_embedding = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=question,
        task_type="retrieval_query",
    )["embedding"]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
    )

    # results["documents"] is a list-of-lists (one inner list per query
    # embedding we sent -- we only sent one, so we take index [0]).
    documents = results.get("documents", [[]])[0]
    return documents


# ---------------------------------------------------------------------------
# STAGE 4b: GENERATE THE FINAL ANSWER WITH GEMINI
# ---------------------------------------------------------------------------

def generate_answer(question: str, context_chunks: list[str]) -> str:
    """
    Sends the retrieved chunks + the user's question to Gemini
    (gemini-3.1-flash-lite) and asks it to answer using ONLY that context.

    WHY THIS PROMPT STYLE ("GROUNDING"):
    This is the "augmented generation" half of RAG. By explicitly
    instructing the model to answer only from the provided context (and to
    say it doesn't know otherwise), we reduce "hallucination" -- the model
    making up a plausible-sounding but false answer (e.g. inventing a price
    for a dish that isn't actually on the menu). The model is still free to
    phrase the answer naturally, but it's constrained to the facts we
    retrieved from the actual uploaded PDF.
    """
    context_text = "\n\n---\n\n".join(context_chunks)

    prompt = f"""You are a helpful assistant answering questions about a restaurant menu.
Answer the question using ONLY the context below, which was extracted from the
restaurant's menu PDF. Do not use any outside knowledge.
If the answer is not contained in the context, say clearly that you don't know
based on the menu provided. Do not guess or make up prices, dish names, or
ingredients that are not in the context.

Context from the menu:
{context_text}

Question: {question}

Answer:"""

    model = genai.GenerativeModel(GENERATION_MODEL)
    response = model.generate_content(prompt)
    return response.text.strip()
