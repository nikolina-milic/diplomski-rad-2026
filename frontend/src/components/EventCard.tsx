import type { Decision } from '../types'
import { ActionBadge, fmtAmount, fmtScore, fmtTime, fullName, place } from './ui'

/**
 * Kompaktan prikaz jedne odluke, za kolone po nivou rizika.
 *
 * Nivo se NE prikazuje bedžom — kolona u kojoj kartica stoji već to kaže.
 * Ponavljati ga na svakoj kartici bilo bi šum.
 */
export default function EventCard({ d, onOpen }: { d: Decision; onOpen: () => void }) {
  const name = fullName(d.event?.first_name ?? d.first_name,
                        d.event?.last_name ?? d.last_name)
  return (
    <button className="evcard" onClick={onOpen}
      aria-label={`Detalji odluke za ${name || d.user_id}, rizik ${fmtScore(d.final_score)}`}>
      <div className="evcard-top">
        <span className="amt">
          {d.event ? fmtAmount(d.event.amount, d.event.currency) : d.user_id}
        </span>
        <span className="sc">{fmtScore(d.final_score)}</span>
      </div>
      <div className="evcard-mid">
        {d.event && <span className="mcat">{d.event.merchant_category}</span>}
        {d.event && <span className="chip">{d.event.channel}</span>}
        {d.event?.home_country && d.event.home_country !== d.event.country && (
          <span className="travel">✈ {d.event.home_country} → {d.event.country}</span>
        )}
      </div>
      <div className="evcard-who">
        {name || d.user_id}{d.event && <> · {place(d.event)}</>}
      </div>
      <div className="evcard-foot">
        <ActionBadge action={d.action} />
        {d.guardrail && <span className="badge badge--brand">guardrail</span>}
        <span className="evcard-time">{d.event && fmtTime(d.event.timestamp)}</span>
      </div>
    </button>
  )
}
