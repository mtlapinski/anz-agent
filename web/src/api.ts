import type { ChatResponse } from "./types";

export async function createSession(provider: string, model: string): Promise<string> {
  const res = await fetch("/session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider, model }),
  });
  const body = await res.json();
  if (!res.ok) {
    throw new Error(body.detail ?? "failed to create session");
  }
  return body.thread_id;
}

export async function sendChat(threadId: string, message: string): Promise<ChatResponse> {
  const res = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ thread_id: threadId, message }),
  });
  return res.json();
}

export async function sendResume(threadId: string, score: number, note?: string): Promise<ChatResponse> {
  const res = await fetch("/resume", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ thread_id: threadId, score, note }),
  });
  return res.json();
}
