import { useEffect, useRef, useState } from "react";

/**
 * ChatArea: the main chat UI. Shows the conversation as bubbles (user on
 * the right, bot on the left), a text input, and a loading indicator while
 * waiting for the backend's answer.
 *
 * This component doesn't call the backend directly — it just collects the
 * user's typed question and hands it up to onSend, then renders whatever
 * messages App gives it back. Keeping the "how do we talk to the API" logic
 * out of this component is what makes it easy to test/reason about on its
 * own.
 */
export default function ChatArea({ messages, onSend, isLoading, documentReady }) {
  const [input, setInput] = useState("");
  const messagesEndRef = useRef(null);

  // Auto-scroll to the newest message whenever the list changes.
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const handleSubmit = (e) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;
    onSend(trimmed);
    setInput("");
  };

  return (
    <main className="chat-area">
      <div className="messages-list">
        {messages.length === 0 && (
          <div className="empty-state">
            <p>
              {documentReady
                ? "Your menu is indexed. Ask a question below to get started."
                : "Upload a menu PDF on the left, then ask questions about it here."}
            </p>
          </div>
        )}

        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`bubble-row ${msg.role === "user" ? "bubble-row-user" : "bubble-row-bot"}`}
          >
            <div className={`bubble ${msg.role === "user" ? "bubble-user" : "bubble-bot"} ${msg.isError ? "bubble-error" : ""}`}>
              {msg.text}
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="bubble-row bubble-row-bot">
            <div className="bubble bubble-bot bubble-loading">
              <span className="loading-dot" />
              <span className="loading-dot" />
              <span className="loading-dot" />
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <form className="chat-input-form" onSubmit={handleSubmit}>
        <input
          type="text"
          className="chat-input"
          placeholder={
            documentReady
              ? "Ask about the menu, e.g. \"what vegetarian starters do you have?\""
              : "Upload a menu PDF first..."
          }
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={isLoading}
        />
        <button
          type="submit"
          className="send-button"
          disabled={isLoading || !input.trim()}
        >
          Send
        </button>
      </form>
    </main>
  );
}
