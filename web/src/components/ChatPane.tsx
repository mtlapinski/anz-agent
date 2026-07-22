import { useState } from "react";
import ReactMarkdown from "react-markdown";
import EvalWidget from "./EvalWidget";
import type { EvalRequestResponse } from "../types";

export interface Message {
  role: "user" | "assistant" | "system";
  text: string;
}

interface Props {
  messages: Message[];
  pendingEval: EvalRequestResponse | null;
  loading: boolean;
  onSend: (message: string) => void;
  onEvalSubmit: (score: number, note?: string) => void;
}

export default function ChatPane({ messages, pendingEval, loading, onSend, onEvalSubmit }: Props) {
  const [input, setInput] = useState("");

  function handleSubmit() {
    if (!input.trim() || loading || pendingEval) return;
    onSend(input.trim());
    setInput("");
  }

  return (
    <div className="chat-pane">
      <div className="pane-header">Chat</div>
      {messages.map((m, i) => (
        <div key={i} className={`message ${m.role}`}>
          {m.role === "user" ? (
            m.text
          ) : (
            <ReactMarkdown>{m.text}</ReactMarkdown>
          )}
        </div>
      ))}
      {pendingEval && <EvalWidget recommendation={pendingEval.recommendation} onSubmit={onEvalSubmit} />}
      <div className="chat-input">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
          disabled={loading || !!pendingEval}
          placeholder="What can I help you find?"
        />
        <button onClick={handleSubmit} disabled={loading || !!pendingEval}>Send</button>
      </div>
    </div>
  );
}
