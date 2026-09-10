import { useEffect, useState } from 'react'
import type { Decision } from '../types'
import { api } from '../api'
import { RiskBadge, ActionBadge, fmtAmount, fmtScore, place, fullName } from '../components/ui'
import DecisionDetail from '../components/DecisionDetail'
import Modal from '../components/Modal'

export default function ReviewQueue() {
  const [items, setItems] = useState<Decision[]>([])
  const [sel, setSel] = useState<Decision | null>(null)
  const [busy, setBusy] = useState<number | null>(null)

  const load = () => api.reviewQueue().then(setItems).catch(() => setItems([]))
  useEffect(() => { load() }, [])

  async function decide(d: Decision, label: number) {
    if (d.id == null) return
    setBusy(d.id)
    await api.feedback(d.id, label)
    setBusy(null)
    if (sel?.id === d.id) setSel(null)
    load()
  }

  return (
    <div>
      <div>
        <div className="section-title">
          <h2>Za pregled</h2>
          <span className="muted" style={{ fontSize: 12 }}>{items.length} slučajeva</span>
        </div>
        <div className="card">
          {items.length === 0 && <div className="empty">Red za pregled je prazan. Analitičar je sve obradio. 🎉</div>}
          {items.map((d) => (
            <div key={d.id} className="tape-row" onClick={() => setSel(d)}>
              <div className="spine" style={{ background: 'var(--med)' }} />
              <div className="txn">
                <div className="l1">
                  <span className="amt">{d.event ? fmtAmount(d.event.amount, d.event.currency) : d.user_id}</span>
                  {d.event && <span className="mcat">{d.event.merchant_category}</span>}
                  {d.event?.home_country && d.event.home_country !== d.event.country && (
                    <span className="travel">✈ {d.event.home_country} → {d.event.country}</span>
                  )}
                </div>
                <div className="l2">
                  {fullName(d.event?.first_name ?? d.first_name, d.event?.last_name ?? d.last_name) && (
                    <><span>{fullName(d.event?.first_name ?? d.first_name, d.event?.last_name ?? d.last_name)}</span><span>·</span></>
                  )}
                  <span>{d.user_id}</span>
                  {d.event && <><span>·</span><span>{place(d.event)}</span></>}
                </div>
              </div>
              <div className="right">
                <div className="row gap" style={{ gap: 8 }}>
                  <RiskBadge level={d.level} />
                  <ActionBadge action={d.action} />
                  <span className="sc">{fmtScore(d.final_score)}</span>
                </div>
                <div className="row gap" style={{ gap: 6 }}>
                  <button className="btn btn--bad" disabled={busy === d.id}
                    onClick={(e) => { e.stopPropagation(); decide(d, 1) }}>Prevara</button>
                  <button className="btn btn--ok" disabled={busy === d.id}
                    onClick={(e) => { e.stopPropagation(); decide(d, 0) }}>Legitimno</button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
      <Modal open={sel !== null} onClose={() => setSel(null)} title="Objašnjenje odluke">
        {sel && <DecisionDetail d={sel} />}
      </Modal>
    </div>
  )
}
