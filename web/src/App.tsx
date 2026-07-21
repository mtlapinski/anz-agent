import { useState } from "react";
import SessionBar from "./components/SessionBar";
import { createSession } from "./api";

export default function App() {
  const [threadId, setThreadId] = useState<string | null>(null);
  const [sessionError, setSessionError] = useState<string | null>(null);

  async function handleStart(provider: string, model: string) {
    setSessionError(null);
    try {
      const id = await createSession(provider, model);
      setThreadId(id);
    } catch (e) {
      setSessionError((e as Error).message);
    }
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
      <p>Session started: {threadId}</p>
    </div>
  );
}
