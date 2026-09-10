import type { Decision } from '../types'
import { ActionBadge, RiskBadge, fmtAmount, fmtScore, fmtTime, fullName, place } from './ui'

function ShapBar({ feature, value, max }: { feature: string; value: number; max: number }) {
  const pct = max > 0 ? Math.abs(value) / max : 0
  const positive = value >= 0
  return (
    <div className="shaprow">
      <span className="mono shaprow-name" title={feature}>{feature}</span>
      <div className="track">
        <span style={{
          left: positive ? '50%' : `${50 - pct * 50}%`,
          width: `${pct * 50}%`,
          background: positive ? 'var(--high)' : 'var(--low)',
        }} />
        <span style={{ left: '50%', width: 1, background: 'var(--border-strong)' }} />
      </div>
      <span className="mono" style={{ textAlign: 'right', color: positive ? 'var(--high)' : 'var(--low)' }}>
        {value >= 0 ? '+' : ''}{value.toFixed(2)}
      </span>
    </div>
  )
}

/**
 * Sadržaj modala sa objašnjenjem jedne odluke. Okvir (naslov, zatvaranje) daje
 * `Modal`, pa se ovdje renderuje samo sadržaj.
 *
 * Raspored prati redoslijed pitanja koja analitičar postavlja:
 *   ko i šta  →  šta je sistem zaključio  →  koliko je siguran  →
 *   činjenice o transakciji | zašto je odlučio tako
 */
export default function DecisionDetail({ d }: { d: Decision }) {
  const shap = d.shap_top ?? []
  const maxAbs = Math.max(0.0001, ...shap.map((s) => Math.abs(s.value)))
  const name = fullName(d.event?.first_name ?? d.first_name,
                        d.event?.last_name ?? d.last_name)
  const rules = d.fired_rules ?? []

  return (
    <div className="detail">
      {/* ---- ko i šta ---- */}
      <header className="dt-head">
        <div>
          {name && <div className="dt-name">{name}</div>}
          <div className="dt-ids mono">{d.user_id} · {d.event_id}</div>
        </div>
        <div className="dt-badges">
          <RiskBadge level={d.level} />
          <ActionBadge action={d.action} />
          {d.guardrail && <span className="badge badge--brand">{d.guardrail}</span>}
        </div>
      </header>

      {d.executed_action && d.executed_action !== d.action && (
        <div className="dt-note">
          Predloženo <strong>{d.action}</strong>, izvršeno <strong>{d.executed_action}</strong>
          {d.mode === 'shadow' && ' — sistem je u shadow režimu, akcija se samo loguje.'}
        </div>
      )}

      {/* ---- sažetak u jednoj rečenici ---- */}
      <div className="explain">{d.explanation}</div>

      {/* ---- četiri broja koja nose odluku ---- */}
      <div className="dt-scores">
        <div className="dt-score dt-score--main">
          <div className="v">{fmtScore(d.final_score, 2)}</div>
          <div className="k">Finalni rizik</div>
        </div>
        <div className="dt-score">
          <div className="v">{d.rules_score != null ? fmtScore(d.rules_score, 1) : '—'}</div>
          <div className="k">Pravila</div>
        </div>
        <div className="dt-score">
          <div className="v">{d.ml_score != null ? fmtScore(d.ml_score, 1) : '—'}</div>
          <div className="k">ML model</div>
        </div>
        <div className="dt-score">
          <div className="v">{d.model_version}</div>
          <div className="k">Verzija modela</div>
        </div>
      </div>

      {/* ---- činjenice | obrazloženje ---- */}
      <div className="dt-cols">
        <section className="dt-col">
          <div className="eyebrow">Transakcija</div>
          {d.event ? (
            <div className="evgrid">
              <span className="k">Iznos</span>
              <span className="v">{fmtAmount(d.event.amount, d.event.currency)}</span>
              <span className="k">Kategorija</span><span className="v">{d.event.merchant_category}</span>
              <span className="k">Kanal</span><span className="v">{d.event.channel}</span>
              <span className="k">Odakle</span><span className="v">{place(d.event)}</span>
              <span className="k">Matična zemlja</span><span className="v">{d.event.home_country ?? '—'}</span>
              <span className="k">Udaljenost od doma</span>
              <span className="v">{Math.round(d.event.distance_from_home_km)} km</span>
              <span className="k">Uređaj</span><span className="v">{d.event.device_id}</span>
              <span className="k">Vrijeme</span><span className="v">{fmtTime(d.event.timestamp)}</span>
            </div>
          ) : (
            <div className="dt-none">Detalji transakcije nisu sačuvani uz ovu odluku.</div>
          )}
        </section>

        <section className="dt-col">
          <div className="eyebrow">Okinuta pravila</div>
          {rules.length > 0 ? (
            rules.map((r) => (
              <div className="rulechip" key={r.name}>
                <span>{r.explanation}</span>
                <span className="mono muted">{r.weight.toFixed(2)}</span>
              </div>
            ))
          ) : (
            <div className="dt-none">Nijedno pravilo nije okinuto.</div>
          )}

          <div className="eyebrow" style={{ marginTop: 16 }}>SHAP doprinosi (ML)</div>
          {shap.length > 0 ? (
            <div className="dt-shap">
              {shap.map((s) => (
                <ShapBar key={s.feature} feature={s.feature} value={s.value} max={maxAbs} />
              ))}
            </div>
          ) : (
            <div className="dt-none">Model ne nudi SHAP objašnjenje.</div>
          )}
        </section>
      </div>
    </div>
  )
}
