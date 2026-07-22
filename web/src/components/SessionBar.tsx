import { useState } from "react";

interface Props {
  onStart: (provider: string, model: string) => void;
}

const DEFAULTS: Record<string, string> = {
  google: "gemini-flash-lite-latest",
  anthropic: "claude-haiku-4-5-20251001",
};

export default function SessionBar({ onStart }: Props) {
  const [provider, setProvider] = useState("google");
  const [model, setModel] = useState(DEFAULTS.google);

  function handleProviderChange(next: string) {
    setProvider(next);
    setModel(DEFAULTS[next]);
  }

  return (
    <div className="session-bar">
      <select value={provider} onChange={(e) => handleProviderChange(e.target.value)}>
        <option value="google">Google</option>
        <option value="anthropic">Anthropic</option>
      </select>
      <input value={model} onChange={(e) => setModel(e.target.value)} placeholder="Model" />
      <button onClick={() => onStart(provider, model)}>Start</button>
    </div>
  );
}
