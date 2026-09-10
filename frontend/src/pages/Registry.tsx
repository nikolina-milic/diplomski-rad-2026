import { useEffect, useState } from 'react'
import type { ModelVersion } from '../types'
import { api } from '../api'

const statusBadge = (s: string) =>
  s === 'champion' ? 'badge--low' : s === 'challenger' ? 'badge--med' : 'badge--ghost'

export default function Registry() {
  const [models, setModels] = useState<ModelVersion[]>([])
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<string | null>(null)

  const load = () => api.models().then(setModels).catch(() => {})
  useEffect(() => { load() }, [])

  async function retrain() {
    setBusy(true); setMsg('Treniram challenger i poredim sa champion-om…')
    try {
      const r = await api.retrain() as {
        promoted: boolean
        challenger_version: string
        champion_metrics: Record<string, number>
        challenger_metrics: Record<string, number>
      }
      setMsg(
        `${r.challenger_version}: PR-AUC ${r.challenger_metrics.pr_auc?.toFixed(3)} vs champion ${r.champion_metrics.pr_auc?.toFixed(3)} — ` +
        (r.promoted ? 'PROMOVISAN u champion ✔' : 'zadržan kao challenger')
      )
    } catch (e) {
      // backend vraća razlog (npr. premalo labeliranih odluka) — prikaži ga
      setMsg(e instanceof Error ? e.message : 'Greška pri retreningu.')
    }
    setBusy(false)
    load()
  }

  return (
    <div>
      <div className="section-title">
        <h2>Verzije modela</h2>
        <button className="btn btn--brand" disabled={busy} onClick={retrain} style={{ marginLeft: 'auto' }}>
          {busy ? 'Treniram…' : 'Pokreni retrening'}
        </button>
      </div>
      {msg && <div className="card pad" style={{ marginBottom: 14, fontSize: 13 }}>{msg}</div>}
      <div className="card">
        <table>
          <thead>
            <tr><th>Verzija</th><th>Status</th><th className="num">PR-AUC</th><th className="num">F1</th><th className="num">Preciznost</th><th className="num">Odziv</th></tr>
          </thead>
          <tbody>
            {models.length === 0 && <tr><td colSpan={6}><div className="empty">Nema registrovanih verzija.</div></td></tr>}
            {models.map((m) => (
              <tr key={m.version}>
                <td className="mono">{m.version}</td>
                <td><span className={`badge ${statusBadge(m.status)}`}>{m.status}</span></td>
                <td className="mono num">{m.metrics?.pr_auc?.toFixed(3) ?? '—'}</td>
                <td className="mono num">{m.metrics?.f1?.toFixed(3) ?? '—'}</td>
                <td className="mono num">{m.metrics?.precision?.toFixed(3) ?? '—'}</td>
                <td className="mono num">{m.metrics?.recall?.toFixed(3) ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
