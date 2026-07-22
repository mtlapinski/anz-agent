import { useState } from "react";

interface Props {
  onSubmit: (score: number, note?: string) => void;
}

export default function EvalWidget({ onSubmit }: Props) {
  const [score, setScore] = useState(5);
  const [note, setNote] = useState("");

  return (
    <div className="eval-widget">
      <label>Rate usefulness (1-5):</label>
      <select value={score} onChange={(e) => setScore(Number(e.target.value))}>
        {[1, 2, 3, 4, 5].map((n) => (
          <option key={n} value={n}>{n}</option>
        ))}
      </select>
      <input placeholder="Note (optional)" value={note} onChange={(e) => setNote(e.target.value)} />
      <button onClick={() => onSubmit(score, note || undefined)}>Submit rating</button>
    </div>
  );
}
