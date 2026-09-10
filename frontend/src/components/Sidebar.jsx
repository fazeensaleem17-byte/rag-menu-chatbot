import { useRef, useState } from "react";

/**
 * Sidebar: lets the user pick a menu PDF and upload it.
 *
 * This is the "ingestion" half of the RAG app — everything here happens
 * once per menu, before any questions get asked. It doesn't know anything
 * about embeddings or chunking itself; it just sends the raw file to the
 * backend (via the onUpload prop) and displays whatever status comes back.
 */
export default function Sidebar({ onUpload, status }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0] ?? null;
    setSelectedFile(file);
  };

  const handleUploadClick = () => {
    if (selectedFile) {
      onUpload(selectedFile);
    }
  };

  const isUploading = status.state === "uploading";

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h1 className="app-title">AskTheMenu</h1>
        <p className="app-subtitle">RAG-powered menu Q&amp;A</p>
      </div>

      <div className="upload-section">
        <label className="section-label" htmlFor="pdf-input">
          1. Upload a menu PDF
        </label>

        <input
          id="pdf-input"
          ref={fileInputRef}
          type="file"
          accept="application/pdf"
          onChange={handleFileChange}
          className="file-input"
        />

        {selectedFile && (
          <p className="selected-filename" title={selectedFile.name}>
            {selectedFile.name}
          </p>
        )}

        <button
          type="button"
          className="upload-button"
          onClick={handleUploadClick}
          disabled={!selectedFile || isUploading}
        >
          {isUploading ? "Uploading..." : "Upload document"}
        </button>

        {status.state !== "idle" && (
          <p className={`status-message status-${status.state}`}>
            {status.message}
          </p>
        )}
      </div>

      <div className="sidebar-footer">
        <p className="hint">
          2. Once indexed, ask questions about the menu in the chat →
        </p>
      </div>
    </aside>
  );
}
