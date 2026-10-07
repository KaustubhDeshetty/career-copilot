# Career Copilot: web front end

A Next.js 15 + React 19 + TypeScript UI for the Career Copilot multi-agent pipeline. The Python agents are unchanged. This app talks to them through a small FastAPI layer (`../api.py`).

```
Browser (React)  ──▶  Next.js route handlers (Node)  ──▶  FastAPI (api.py)  ──▶  run_pipeline()
  live agent timeline     zod validation, rate limit,        SSE progress stream     Router ∥ Planner → Selector →
  fit panel, quiz cards   hides the backend URL              tracker endpoints       Researcher → Writer ⇄ Critic
```

## Features

- **Live agent timeline.** Each agent (Router, Planner, Selector, Researcher, and Writer ⇄ Critic by round) appears as it starts and finishes, streamed over Server-Sent Events.
- **Apply mode.** A requirements-met score, a verdict pill, a role-archetype ranking chart, and the full fit table rendered from the LLM's Markdown.
- **Interview mode.** Quiz cards with show/hide answers.
- **Learn mode.** A revision note with its sources.
- **Tracker page.** A sortable view of `applications.xlsx`. Saves from the web app and from Streamlit go to the same workbook, with the same duplicate guard.
- **Node layer as a backend-for-frontend.** Requests are validated with zod (length caps, types), rate-limited, and proxied, so the browser never talks to Python directly.
- **Shared types.** `lib/types.ts` mirrors the Python `PipelineState` contract.

## Run it locally (two terminals)

**Terminal 1: Python backend**, from the Career Copilot folder:

```powershell
pip install fastapi "uvicorn[standard]"
$env:GROQ_API_KEY = "your-key"
uvicorn api:app --port 8000
```

**Terminal 2: web front end**, from this `web` folder:

```powershell
npm install
npm run dev
```

Open http://localhost:3000. The status dot in the top bar shows whether the backend is reachable and whether `GROQ_API_KEY` is set.

To point the UI at a backend somewhere else, copy `.env.example` to `.env.local` and change `BACKEND_URL`.

## Scripts

| Command | What it does |
|---|---|
| `npm run dev` | Dev server with hot reload |
| `npm run build` then `npm start` | Production build |
| `npm run typecheck` | TypeScript check only |

## Project structure

```
app/
  page.tsx                      assistant: input, live timeline, results
  tracker/page.tsx              sortable tracker tables
  api/run/route.ts              validate → proxy → stream SSE back
  api/tracker/...               tracker read / save proxies
  api/health/route.ts           backend + API-key status
components/                     AgentTimeline, FitPanel, QuizCards, ResultView, Markdown, BackendStatus
lib/
  types.ts                      shared API contract (mirrors state.py)
  sse.ts                        fetch-based SSE reader (EventSource can't POST)
  validation.ts                 zod schemas
  rateLimit.ts                  in-memory limiter for /api/run
```
