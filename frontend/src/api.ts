import type {
  Calibration, CostConfig, CostMatrix, Capacity, Decision, DecisionCount, Drift,
  Evaluation, Fusion, Level, Metrics, ModelVersion, Policy, Rule, Stats,
} from './types'

const BASE = ((import.meta.env as Record<string, string>).VITE_API_BASE) || 'http://localhost:8000'
const WS_BASE = BASE.replace(/^http/, 'ws')

async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(BASE + path)
  if (!r.ok) throw new Error(`GET ${path} -> ${r.status}`)
  return r.json()
}

async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(BASE + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) {
    // backend objašnjava zašto (npr. premalo labeliranih odluka za retrening)
    const detail = await r.json().catch(() => null)
    throw new Error(detail?.detail || `POST ${path} -> ${r.status}`)
  }
  return r.json()
}

async function putJSON<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(BASE + path, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) {
    const detail = await r.json().catch(() => null)
    throw new Error(detail?.detail || `PUT ${path} -> ${r.status}`)
  }
  return r.json()
}

export const api = {
  decisions: (limit = 50, level?: Level) =>
    getJSON<Decision[]>(`/decisions?limit=${limit}${level ? `&level=${level}` : ''}`),
  decisionsCount: () => getJSON<DecisionCount>(`/decisions/count`),
  fusion: () => getJSON<Fusion>(`/fusion`),
  updatePolicy: (b: Partial<Policy>) => putJSON<Policy>(`/config/policy`, b),
  updateFusion: (b: Partial<Fusion>) => putJSON<Fusion>(`/config/fusion`, b),
  updateRules: (updates: { name: string; weight?: number; enabled?: boolean; thresholds?: number[] }[]) =>
    putJSON<Rule[]>(`/config/rules`, updates),
  reviewQueue: () => getJSON<Decision[]>(`/review-queue`),
  feedback: (decision_id: number, label: number) =>
    postJSON(`/feedback`, { decision_id, label, source: 'analyst' }),
  models: () => getJSON<ModelVersion[]>(`/models`),
  stats: () => getJSON<Stats>(`/stats`),
  metrics: () => getJSON<Metrics>(`/metrics`),
  evaluation: (nTest = 800) => getJSON<Evaluation>(`/evaluation?n_test=${nTest}`),
  rules: () => getJSON<Rule[]>(`/rules`),
  policy: () => getJSON<Policy>(`/policy`),
  // Verziju bira backend — frontend je ranije slao `v{n+1}` i na praznom
  // registryju tražio "v1", pregazivši artefakt početnog modela.
  retrain: () => postJSON(`/retrain`, {}),
  cost: () => getJSON<CostConfig>(`/cost`),
  updateCost: (b: Partial<CostMatrix & Capacity> & { apply?: boolean }) =>
    putJSON<CostConfig>(`/config/cost`, b),
  calibration: () => getJSON<Calibration>(`/calibration`),
  drift: (window = 500) => getJSON<Drift>(`/drift?window=${window}`),
}

// Razmak između događaja u demo toku. Serverski podrazumijevanih 5 s je
// predugo da se ekran napuni pri prikazu.
const STREAM_INTERVAL_S = 2

export function openStream(onMsg: (d: Decision) => void): WebSocket {
  const ws = new WebSocket(`${WS_BASE}/ws/stream?interval=${STREAM_INTERVAL_S}`)
  ws.onmessage = (e) => onMsg(JSON.parse(e.data) as Decision)
  return ws
}
