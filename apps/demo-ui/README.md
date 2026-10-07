# demo-ui

Browser UI for the zone demo. It talks to the FastAPI backend (`uv run pv-demo`) through a
`/api` proxy configured in `next.config.ts`. See the repository README for the full run order.

```bash
npm install
npm run dev     # http://localhost:3000
```

Set `PV_API_URL` if the backend is not on `http://127.0.0.1:8000`.
