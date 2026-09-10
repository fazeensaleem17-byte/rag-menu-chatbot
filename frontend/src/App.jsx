import { useState } from "react";
import Sidebar from "./components/Sidebar.jsx";
import ChatArea from "./components/ChatArea.jsx";
import { uploadMenu, askQuestion } from "./api.js";
import "./App.css";

let nextMessageId = 1;

export default function App() {
  // uploadStatus.state is one of: "idle" | "uploading" | "success" | "error"
  const [uploadStatus, setUploadStatus] = useState({ state: "idle", message: "" });
  const [documentReady, setDocumentReady] = useState(false);
  const [messages, setMessages] = useState([]);
  const [isAsking, setIsAsking] = useState(false);

  const addMessage = (role, text, isError = false) => {
    setMessages((prev) => [...prev, { id: nextMessageId++, role, text, isError }]);
  };

  const handleUpload = async (file) => {
    setUploadStatus({ state: "uploading", message: `Indexing ${file.name}...` });
    try {
      const result = await uploadMenu(file);
      setUploadStatus({
        state: "success",
        message: `"${result.filename}" indexed — ${result.chunks_created} chunks created.`,
      });
      setDocumentReady(true);
    } catch (err) {
      setUploadStatus({ state: "error", message: err.message });
      setDocumentReady(false);
    }
  };

  const handleSend = async (question) => {
    addMessage("user", question);
    setIsAsking(true);
    try {
      const result = await askQuestion(question);
      addMessage("bot", result.answer);
    } catch (err) {
      addMessage("bot", err.message, true);
    } finally {
      setIsAsking(false);
    }
  };

  return (
    <div className="app">
      <Sidebar onUpload={handleUpload} status={uploadStatus} />
      <ChatArea
        messages={messages}
        onSend={handleSend}
        isLoading={isAsking}
        documentReady={documentReady}
      />
    </div>
  );
}
