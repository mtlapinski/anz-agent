import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    // Bind to all interfaces (not just localhost) so the dev server is
    // reachable from another device on the local network. Proxy targets
    // stay 127.0.0.1 — that's the Vite server's own machine talking to
    // its local backend, regardless of which device the browser is on.
    host: true,
    proxy: {
      "/session": "http://127.0.0.1:8000",
      "/chat": "http://127.0.0.1:8000",
      "/resume": "http://127.0.0.1:8000",
    },
  },
});
