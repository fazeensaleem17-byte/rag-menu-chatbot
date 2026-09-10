// api.js
// -----
// All communication with the AskTheMenu backend lives in this one file, so
// the rest of the app never has to know about fetch(), URLs, or response
// shapes directly.

const API_BASE = "http://localhost:8000";

/**
 * Uploads a PDF file to the backend, which extracts its text, chunks it,
 * embeds each chunk, and stores it in ChromaDB (see /upload in main.py).
 *
 * Returns: { filename, chunks_created }
 * Throws: an Error with a human-readable message on failure.
 */
export async function uploadMenu(file) {
  const formData = new FormData();
  formData.append("file", file);

  let response;
  try {
    response = await fetch(`${API_BASE}/upload`, {
      method: "POST",
      body: formData,
    });
  } catch (networkError) {
    throw new Error(
      "Could not reach the backend. Is it running at http://localhost:8000?"
    );
  }

  const data = await safeParseJson(response);

  if (!response.ok) {
    throw new Error(data?.detail || `Upload failed (status ${response.status}).`);
  }

  return data;
}

/**
 * Sends a question to the backend, which retrieves the most relevant menu
 * chunks and asks Gemini to answer from them (see /query in main.py).
 *
 * Returns: { answer }
 * Throws: an Error with a human-readable message on failure.
 */
export async function askQuestion(question) {
  let response;
  try {
    response = await fetch(`${API_BASE}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
  } catch (networkError) {
    throw new Error(
      "Could not reach the backend. Is it running at http://localhost:8000?"
    );
  }

  const data = await safeParseJson(response);

  if (!response.ok) {
    throw new Error(data?.detail || `Query failed (status ${response.status}).`);
  }

  return data;
}

// Some error responses might not be valid JSON (e.g. a proxy error page).
// This helper avoids a confusing second error if that happens.
async function safeParseJson(response) {
  try {
    return await response.json();
  } catch {
    return null;
  }
}
