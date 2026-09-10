import { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  LineChart, Line, Legend,
} from 'recharts'
import type { Metrics as M, Evaluation, Calibration, Drift } from '../types'
import { api } from '../api'
import { fmtScore } from '../components/ui'

const pct = (v?: number) => (v == null ? '—' : fmtScore(v, 1))

const APPROACH_LABELS: Record<string, string> = {
  rules: 'Samo pravila',
  ml: 'Samo ML',
  hybrid_weighted: 'Hibrid (weighted)',
  hybrid_cascade: 'Hibrid (cascade)',
  hybrid_stacking: 'Hibrid (stacking)',
}

const SEVERITY_COLOR: Record<string, string> = {
  'stabilno': 'var(--low)', 'umjereno': 'var(--med)', 'značajno': 'var(--high)',
}

// Boje grafikona prate tokene iz index.css. Recharts ne razumije var(),
// pa se vrijednosti drže ovdje na jednom mjestu.
const C = {
  grid: '#dde5ea',
  axis: '#7c8f9b',
  accent: '#1d5c72',
  accentSoft: '#7aa8bd',
  reference: '#a8c2d0',
} as const

const AXIS = { stroke: C.axis, fontSize: 12, tickLine: false } as const
const TOOLTIP = {
  background: '#ffffff',
  border: '1px solid #c3d0d8',
  borderRadius: 6,
  boxShadow: '0 8px 24px rgba(20,40,55,0.14)',
  fontFamily: "'IBM Plex Mono', monospace",
  fontSize: 12,
  color: '#141c22',
} as const

export default function Metrics() {
  const [m, setM] = useState<M | null>(null)
  const [ev, setEv] = useState<Evaluation | null>(null)
  const [evLoading, setEvLoading] = useState(true)
  const [cal, setCal] = useState<Calibration | null>(null)
  const [drift, setDrift] = useState<Drift | null>(null)

  useEffect(() => {
    const load = () => api.metrics().then(setM).catch(() => {})
    load()
    const t = setInterval(load, 4000)
    api.evaluation(800).then(setEv).catch(() => {}).finally(() => setEvLoading(false))
    api.calibration().then(setCal).catch(() => {})
    api.drift().then(setDrift).catch(() => {})
    return () => clearInterval(t)
  }, [])

  return (
    <div>
      {/* ---------- Uživo iz logovanih odluka ---------- */}
      <div className="section-title"><h2>Uživo (na logovanim odlukama)</h2>
        <span className="muted" style={{ fontSize: 12 }}>{m?.n ?? 0} označenih događaja</span></div>
      <div className="tiles">
        {[['PR-AUC', m?.live_metrics?.pr_auc], ['F1', m?.live_metrics?.f1],
          ['Preciznost', m?.live_metrics?.precision], ['Odziv', m?.live_metrics?.recall]].map(([k, v]) => (
          <div className="card tile" key={k as string}>
            <div className="v">{pct(v as number)}</div>
            <div className="k">{k as string}</div>
          </div>
        ))}
      </div>

      {m && (
        <>
          <div className="section-title"><h2>Konfuziona matrica</h2>
            <span className="muted" style={{ fontSize: 12 }}>označeno = akcija BLOCK/REVIEW</span></div>
          <div className="confusion">
            {[
              { cn: m.confusion.tp, cl: 'TP · uhvaćena prevara', color: 'var(--low)' },
              { cn: m.confusion.fp, cl: 'FP · lažna uzbuna', color: 'var(--med)' },
              { cn: m.confusion.fn, cl: 'FN · propuštena prevara', color: 'var(--high)' },
              { cn: m.confusion.tn, cl: 'TN · ispravno propušteno', color: 'var(--faint)' },
            ].map((x) => (
              <div className="card conf-cell" key={x.cl}>
                <div className="cn" style={{ color: x.color }}>{x.cn}</div>
                <div className="cl">{x.cl}</div>
              </div>
            ))}
          </div>
        </>
      )}

      {/* ---------- Evaluacija: pravila vs ML vs hibrid ---------- */}
      <div className="section-title"><h2>Pravila vs ML vs Hibrid</h2>
        <span className="muted" style={{ fontSize: 12 }}>
          {evLoading ? 'računam na nezavisnom test setu…' : ev ? `${ev.n_test} događaja · ${ev.n_fraud} prevara` : ''}
        </span></div>

      {!ev ? (
        <div className="card pad empty">{evLoading ? 'Evaluacija u toku (par sekundi)…' : 'Evaluacija nedostupna.'}</div>
      ) : (
        <>
          <div className="card pad" style={{ height: 300 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={Object.entries(ev.approaches).map(([k, v]) => ({
                  name: APPROACH_LABELS[k] ?? k,
                  'PR-AUC': Number(v.metrics.pr_auc.toFixed(3)),
                  F1: Number(v.metrics.f1.toFixed(3)),
                }))}
                margin={{ top: 10, right: 10, left: -18, bottom: 0 }}
              >
                <CartesianGrid stroke={C.grid} vertical={false} />
                <XAxis dataKey="name" {...AXIS} />
                <YAxis domain={[0, 1]} {...AXIS} />
                <Tooltip contentStyle={TOOLTIP} cursor={{ fill: '#ffffff08' }} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="PR-AUC" fill={C.accent} radius={[3, 3, 0, 0]} isAnimationActive={false} />
                <Bar dataKey="F1" fill={C.accentSoft} radius={[3, 3, 0, 0]} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="section-title"><h2>PR kriva — hibrid (stacking)</h2></div>
          <div className="card pad" style={{ height: 280 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={(ev.approaches.hybrid_stacking ?? ev.approaches.hybrid_weighted).pr_curve}
                margin={{ top: 10, right: 16, left: -18, bottom: 12 }}>
                <CartesianGrid stroke={C.grid} />
                <XAxis dataKey="recall" type="number" domain={[0, 1]} {...AXIS}
                  label={{ value: 'Recall', position: 'insideBottom', offset: -2, fill: C.axis, fontSize: 11 }} />
                <YAxis dataKey="precision" domain={[0, 1]} {...AXIS} />
                <Tooltip contentStyle={TOOLTIP} />
                <Line type="monotone" dataKey="precision" stroke={C.accent}
                  strokeWidth={2} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>


          {/* ---------- Kalibracija ---------- */}
          <div className="section-title"><h2>Kalibracija skora</h2>
            <span className="muted" style={{ fontSize: 12 }}>
              niže je bolje · Brier = kvadratna greška vjerovatnoće, ECE = odstupanje od stvarne učestalosti
            </span></div>
          <div className="card">
            <table>
              <thead><tr><th>Pristup</th><th className="num">Brier</th><th className="num">ECE</th></tr></thead>
              <tbody>
                {Object.entries(ev.approaches).map(([k, v]) => (
                  <tr key={k}>
                    <td>{APPROACH_LABELS[k] ?? k}</td>
                    <td className="mono num">{v.calibration ? v.calibration.brier.toFixed(4) : '—'}</td>
                    <td className="mono num">{v.calibration ? v.calibration.ece.toFixed(4) : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {ev.approaches.hybrid_stacking?.calibration && (
            <>
              <div className="section-title"><h2>Reliability dijagram — hibrid (stacking)</h2>
                <span className="muted" style={{ fontSize: 12 }}>
                  savršena kalibracija leži na dijagonali
                </span></div>
              <div className="card pad" style={{ height: 280 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={ev.approaches.hybrid_stacking.calibration.reliability.map((b) => ({
                      predviđeno: Number(b.mean_predicted.toFixed(3)),
                      stvarno: Number(b.fraction_positive.toFixed(3)),
                      idealno: Number(b.mean_predicted.toFixed(3)),
                    }))}
                    margin={{ top: 10, right: 16, left: -18, bottom: 0 }}>
                    <CartesianGrid stroke={C.grid} />
                    <XAxis dataKey="predviđeno" type="number" domain={[0, 1]} {...AXIS}
                      label={{ value: 'Predviđena vjerovatnoća', position: 'insideBottom', offset: -2, fill: C.axis, fontSize: 11 }} />
                    <YAxis domain={[0, 1]} {...AXIS} />
                    <Tooltip contentStyle={TOOLTIP} />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Line type="monotone" dataKey="stvarno" stroke={C.accent} strokeWidth={2} dot
                      isAnimationActive={false} />
                    <Line type="monotone" dataKey="idealno" stroke={C.reference} strokeWidth={1}
                      strokeDasharray="4 4" dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </>
          )}

          {cal?.meta_weights && (
            <div className="card pad" style={{ marginBottom: 14, fontSize: 12.5 }}>
              <strong>Naučene težine fusion sloja</strong> ({cal.method}, strategija {cal.strategy}):{' '}
              <span className="mono">
                pravila {cal.meta_weights.rules >= 0 ? '+' : ''}{cal.meta_weights.rules.toFixed(2)}
                {'  ·  '}ML {cal.meta_weights.ml >= 0 ? '+' : ''}{cal.meta_weights.ml.toFixed(2)}
                {'  ·  '}interakcija {cal.meta_weights.interaction >= 0 ? '+' : ''}{cal.meta_weights.interaction.toFixed(2)}
              </span>
              <div className="muted" style={{ marginTop: 6 }}>
                Meta-model iz podataka uči koliko vjerovati kom izvoru — umjesto fiksnog odnosa 50:50
                nad skorovima koji nisu na istoj skali.
              </div>
            </div>
          )}

          {/* ---------- Trošak politike ---------- */}
          {ev.cost && (
            <>
              <div className="section-title"><h2>Trošak politike</h2>
                <span className="muted" style={{ fontSize: 12 }}>
                  ukupan trošak na test setu, po matrici troška iz banking pack-a
                </span></div>
              <div className="card">
                <table>
                  <thead><tr>
                    <th>Pragovi</th><th className="num">NIZAK do</th><th className="num">SREDNJI do</th>
                    <th className="num">ukupno (EUR)</th><th className="num">po događaju</th>
                  </tr></thead>
                  <tbody>
                    {([['current', 'trenutni'], ['closed_form', 'zatvorena forma'],
                       ['constrained', 'uz kapacitet'], ['empirical', 'empirijski optimum']] as const)
                      .map(([key, label]) => {
                        const d = ev.cost![key]
                        if (!d) return null
                        return (
                          <tr key={key}>
                            <td>{label}</td>
                            <td className="mono num">{fmtScore(d.low_max, 2)}</td>
                            <td className="mono num">{fmtScore(d.medium_max, 2)}</td>
                            <td className="mono num">{d.total_cost.toFixed(0)}</td>
                            <td className="mono num">{d.cost_per_event.toFixed(3)}</td>
                          </tr>
                        )
                      })}
                    <tr>
                      <td className="muted">bez ikakve provjere</td>
                      <td className="mono num">—</td><td className="mono num">—</td>
                      <td className="mono num">{ev.cost.allow_all_baseline.total_cost.toFixed(0)}</td>
                      <td className="mono num">—</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </>
          )}

          {/* ---------- Drift ---------- */}
          <div className="section-title"><h2>Pomjeranje raspodjele (drift)</h2>
            <span className="muted" style={{ fontSize: 12 }}>
              PSI &lt; 0.10 stabilno · 0.10–0.25 umjereno · &gt; 0.25 značajno
            </span></div>
          {!drift?.available ? (
            <div className="card pad empty">
              {drift?.reason ?? 'Nema podataka o driftu.'}
            </div>
          ) : (
            <div className="card">
              <div className="pad" style={{ fontSize: 12.5, borderBottom: '1px solid var(--border)' }}>
                Ukupna ocjena:{' '}
                <span style={{ color: SEVERITY_COLOR[drift.overall ?? ''] ?? 'inherit', fontWeight: 600 }}>
                  {drift.overall}
                </span>
                {'  ·  '}{drift.n_drifting}/{drift.features?.length} obilježja pomjereno
                {'  ·  '}poređeno {drift.n_current} najnovijih sa {drift.n_reference} ranijih odluka
              </div>
              <table>
                <thead><tr>
                  <th>Obilježje</th><th className="num">PSI</th><th>ocjena</th>
                  <th className="num">KS p</th><th className="num">ref. sred.</th><th className="num">tek. sred.</th>
                </tr></thead>
                <tbody>
                  {(drift.features ?? []).slice(0, 8).map((f) => (
                    <tr key={f.feature}>
                      <td className="mono">{f.feature}</td>
                      <td className="mono num">{f.psi.toFixed(3)}</td>
                      <td><span style={{ color: SEVERITY_COLOR[f.severity] ?? 'inherit' }}>{f.severity}</span></td>
                      <td className="mono num">{f.ks_pvalue.toFixed(3)}</td>
                      <td className="mono num">{f.reference_mean.toFixed(2)}</td>
                      <td className="mono num">{f.current_mean.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <div className="section-title"><h2>Latencija (pun engine + SHAP)</h2></div>
          <div className="tiles">
            {[['mean', ev.latency.mean_ms], ['p50', ev.latency.p50_ms],
              ['p95', ev.latency.p95_ms]].map(([k, v]) => (
              <div className="card tile" key={k as string}>
                <div className="v">{(v as number).toFixed(1)}<span style={{ fontSize: 14, color: 'var(--muted)' }}> ms</span></div>
                <div className="k">{k as string} · po događaju</div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
