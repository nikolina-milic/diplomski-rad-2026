import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import type { Decision, DecisionCount, Level } from '../types'
import { api } from '../api'
import {
  ActionBadge, LEVEL_LABEL, RiskBadge, fmtAmount, fmtScore, fmtTime, fullName,
  levelColor, place,
} from '../components/ui'
import DecisionDetail from '../components/DecisionDetail'
import Modal from '../components/Modal'

const LEVELS: Level[] = ['LOW', 'MEDIUM', 'HIGH']
const PAGE_SIZES = [50, 200, 1000]

function isLevel(v: string | null): v is Level {
  return v === 'LOW' || v === 'MEDIUM' || v === 'HIGH'
}

export default function Events() {
  // Filter živi u URL-u, pa je pogled iz „Pogledaj sve" moguće podijeliti
  // linkom i vratiti se na njega dugmetom nazad.
  const [params, setParams] = useSearchParams()
  const raw = params.get('level')
  const level = isLevel(raw) ? raw : null

  const [items, setItems] = useState<Decision[]>([])
  const [counts, setCounts] = useState<DecisionCount | null>(null)
  const [limit, setLimit] = useState(PAGE_SIZES[0])
  const [sel, setSel] = useState<Decision | null>(null)
  const [loading, setLoading] = useState(true)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    api.decisions(limit, level ?? undefined)
      .then((rows) => { setItems(rows); setErr(null) })
      .catch((e) => { setItems([]); setErr(e instanceof Error ? e.message : 'nepoznata greška') })
      .finally(() => setLoading(false))
  }, [level, limit])

  useEffect(() => { api.decisionsCount().then(setCounts).catch(() => {}) }, [])

  const setLevel = (next: Level | null) => {
    if (next) setParams({ level: next })
    else setParams({})
  }

  const total = level ? counts?.by_level?.[level] : counts?.total

  return (
    <div>
      {err && (
        <div className="banner banner--err">Ne mogu da učitam odluke: {err}</div>
      )}
      <div className="card pad filters">
        <div className="filter-group" role="group" aria-label="Filter po nivou rizika">
          <button className={`btn btn--sm ${level === null ? 'on' : ''}`}
            onClick={() => setLevel(null)}>
            Svi{counts ? ` (${counts.total})` : ''}
          </button>
          {LEVELS.map((l) => (
            <button key={l} className={`btn btn--sm ${level === l ? 'on' : ''}`}
              onClick={() => setLevel(l)}>
              {LEVEL_LABEL[l]}{counts ? ` (${counts.by_level?.[l] ?? 0})` : ''}
            </button>
          ))}
        </div>
        <div className="filter-right">
          <span className="muted" style={{ fontSize: 12.5 }}>
            {loading ? 'učitavam…'
              : `prikazano ${items.length}${total != null ? ` od ${total}` : ''}`}
          </span>
          <label className="filter-limit">
            Prikaži
            <select value={limit} onChange={(e) => setLimit(Number(e.target.value))}>
              {PAGE_SIZES.map((n) => <option key={n} value={n}>{n}</option>)}
            </select>
            najnovijih
          </label>
        </div>
      </div>

      <div className="card" style={{ overflowX: 'auto' }}>
        <table className="events">
          <thead>
            <tr>
              <th></th>
              <th>Vrijeme</th>
              <th className="num">Iznos</th>
              <th>Kategorija</th>
              <th>Korisnik</th>
              <th>Odakle</th>
              <th>Nivo</th>
              <th>Akcija</th>
              <th className="num">Rizik</th>
            </tr>
          </thead>
          <tbody>
            {!loading && items.length === 0 && (
              <tr><td colSpan={9}><div className="empty">
                {err ? 'Učitavanje nije uspjelo.' : 'Nema odluka za ovaj filter.'}
              </div></td></tr>
            )}
            {items.map((d, i) => (
              <tr key={`${d.event_id}-${d.id ?? i}`} className="evrow"
                onClick={() => setSel(d)} tabIndex={0}
                onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setSel(d) } }}>
                <td className="evrow-spine">
                  <span style={{ background: levelColor(d.level) }} />
                </td>
                <td>{d.event ? fmtTime(d.event.timestamp) : '—'}</td>
                <td className="mono num">
                  {d.event ? fmtAmount(d.event.amount, d.event.currency) : '—'}
                </td>
                <td>{d.event?.merchant_category ?? '—'}</td>
                <td>
                  {fullName(d.event?.first_name ?? d.first_name,
                            d.event?.last_name ?? d.last_name) || d.user_id}
                </td>
                <td>{d.event ? place(d.event) : '—'}</td>
                <td><RiskBadge level={d.level} /></td>
                <td><ActionBadge action={d.action} /></td>
                <td className="mono num">{fmtScore(d.final_score)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <Modal open={sel !== null} onClose={() => setSel(null)} title="Objašnjenje odluke">
        {sel && <DecisionDetail d={sel} />}
      </Modal>
    </div>
  )
}
