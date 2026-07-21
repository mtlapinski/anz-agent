import { useState } from "react";
import SessionBar from "./components/SessionBar";
import ChatPane, { type Message } from "./components/ChatPane";
import { createSession, sendChat, sendResume } from "./api";
import type { ChatResponse, EvalRequestResponse, Product, ViewType } from "./types";

export default function App() {
  const [threadId, setThreadId] = useState<string | null>(null);
  const [sessionError, setSessionError] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [pendingEval, setPendingEval] = useState<EvalRequestResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [lastProducts, setLastProducts] = useState<Product[] | null>(null);
  const [lastView, setLastView] = useState<ViewType | null>(null);

  async function handleStart(provider: string, model: string) {
    setSessionError(null);
    try {
      const id = await createSession(provider, model);
      setThreadId(id);
    } catch (e) {
      setSessionError((e as Error).message);
    }
  }

  function appendMessage(role: Message["role"], text: string) {
    setMessages((prev) => [...prev, { role, text }]);
  }

  function applyResults(response: { products: Product[] | null; view: ViewType | null }) {
    if (response.products) {
      setLastProducts(response.products);
      setLastView(response.view ?? "cards");
    }
  }

  async function handleSend(message: string) {
    if (!threadId) return;
    appendMessage("user", message);
    setLoading(true);
    const response: ChatResponse = await sendChat(threadId, message);
    if (response.type === "error") {
      appendMessage("system", `Error: ${response.message}`);
    } else if (response.type === "message") {
      appendMessage("assistant", response.text);
      applyResults(response);
    } else {
      appendMessage("assistant", response.recommendation);
      applyResults(response);
      setPendingEval(response);
    }
    setLoading(false);
  }

  async function handleEvalSubmit(score: number, note?: string) {
    if (!threadId) return;
    const response = await sendResume(threadId, score, note);
    setPendingEval(null);
    if (response.type === "message") appendMessage("system", response.text);
    if (response.type === "error") appendMessage("system", `Error: ${response.message}`);
  }

  if (!threadId) {
    return (
      <div className="app">
        <div>
          <h1>Amazon Shopping Agent</h1>
          <SessionBar onStart={handleStart} />
          {sessionError && <p style={{ color: "red" }}>{sessionError}</p>}
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <ChatPane
        messages={messages}
        pendingEval={pendingEval}
        loading={loading}
        onSend={handleSend}
        onEvalSubmit={handleEvalSubmit}
      />
      <div className="results-panel">
        {lastProducts ? (
          <p>{lastProducts.length} results ({lastView ?? "cards"} view) — ResultsPanel component added in the next task</p>
        ) : (
          <p>No search results yet.</p>
        )}
      </div>
    </div>
  );
}
