import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import type { Decision, DecisionCount, Level } from '../types'
import { api, openStream } from '../api'
import { LEVEL_LABEL, levelClass } from '../components/ui'
import DecisionDetail from '../components/DecisionDetail'
import EventCard from '../components/EventCard'
import Modal from '../components/Modal'

/** Koliko odluka po nivou stoji u koloni. Ostalo je na stranici „Svi događaji". */
const PER_COLUMN = 20

const COLUMNS: { level: Level; hint: string }[] = [
  { level: 'LOW', hint: 'propušteno bez smetnje' },
  { level: 'MEDIUM', hint: 'traženo potvrđivanje' },
  { level: 'HIGH', hint: 'ide na ručni pregled' },
]

export default function LiveMonitor() {
  const [feed, setFeed] = useState<Record<Level, Decision[]>>({
    LOW: [], MEDIUM: [], HIGH: [],
  })
  const [sel, setSel] = useState<Decision | null>(null)
  const [counts, setCounts] = useState<DecisionCount | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [live, setLive] = useState(false)

  const push = useCallback((d: Decision) => {
    setFeed((prev) => ({
      ...prev,
      [d.level]: [d, ...(prev[d.level] ?? [])].slice(0, PER_COLUMN),
    }))
  }, [])

  useEffect(() => {
    // Kolone se prvo pune iz audit loga — inače je ekran prazan pri otvaranju
    // i puni se tek kako stižu živi događaji.
    Promise.all(COLUMNS.map((c) => api.decisions(PER_COLUMN, c.level)))
      .then((lists) => setFeed((prev) => {
        const seeded = { ...prev }
        COLUMNS.forEach((c, i) => {
          if (seeded[c.level].length === 0) seeded[c.level] = lists[i]
        })
        return seeded
      }))
      .catch(() => {})

    const ws = openStream((d) => { setLive(true); push(d) })
    ws.onclose = () => setLive(false)

    // Jedan izvor za brojače i za zaglavlja kolona; /stats broji samo
    // posljednjih N odluka, pa bi davao brojke koje se ne slažu sa kolonama.
    const refresh = () => {
      api.decisionsCount()
        .then((c) => { setCounts(c); setErr(null) })
        .catch((e) => setErr(e instanceof Error ? e.message : 'nepoznata greška'))
    }
    refresh()
    const poll = setInterval(refresh, 3000)
    return () => { ws.close(); clearInterval(poll) }
  }, [push])

  const actions = counts?.by_action
  // Bez podataka se prikazuje '—', a ne 0: nula znači „nema takvih odluka",
  // a neuspio zahtjev nešto sasvim drugo. Prikazivati oboje isto je laž.
  const num = (k: string) => (actions ? actions[k] ?? 0 : '—')
  const cells = [
    { l: 'Propušteno', v: num('ALLOW'), c: 'var(--low)' },
    { l: 'Traženo potvrđivanje', v: num('CHALLENGE'), c: 'var(--med)' },
    { l: 'Za ručni pregled', v: num('REVIEW'), c: 'var(--med)' },
    { l: 'Blokirano', v: num('BLOCK'), c: 'var(--high)' },
  ]

  return (
    <div>
      {err && (
        <div className="banner banner--err">
          Backend ne odgovara na <code>/decisions/count</code> ({err}). Ako je
          pokrenut sa starijom verzijom koda, restartuj ga.
        </div>
      )}
      <div className="pulse">
        {cells.map((c) => (
          <div className="card cell" key={c.l}>
            <div className="n" style={{ color: c.c }}>{c.v}</div>
            <div className="l">{c.l}</div>
          </div>
        ))}
      </div>

      <div className="section-title">
        <span className={live ? 'dotlive' : ''}
          style={{ background: live ? undefined : 'var(--faint)' }} />
        <h2>Tok odluka po nivou rizika</h2>
        <span className="muted" style={{ fontSize: 12 }}>
          {live ? 'uživo' : 'povezivanje…'} · po {PER_COLUMN} najnovijih u koloni
        </span>
      </div>

      <div className="lanes">
        {COLUMNS.map(({ level, hint }) => {
          const items = feed[level]
          const total = counts?.by_level?.[level]
          return (
            <section className="lane" key={level} aria-label={`Rizik ${LEVEL_LABEL[level]}`}>
              <header className={`lane-head lane-head--${levelClass(level)}`}>
                <div>
                  <h3>{LEVEL_LABEL[level]}</h3>
                  <span className="lane-hint">{hint}</span>
                </div>
                <div className="lane-meta">
                  <span className="lane-count">{total ?? '—'}</span>
                  <Link className="btn btn--sm" to={`/dogadjaji?level=${level}`}>
                    Pogledaj sve
                  </Link>
                </div>
              </header>
              <div className="lane-body">
                {items.length === 0 && (
                  <div className="empty">Još nema odluka na ovom nivou.</div>
                )}
                {items.map((d, i) => (
                  <EventCard key={`${d.event_id}-${d.id ?? i}`} d={d}
                    onOpen={() => setSel(d)} />
                ))}
              </div>
            </section>
          )
        })}
      </div>

      <Modal open={sel !== null} onClose={() => setSel(null)} title="Objašnjenje odluke">
        {sel && <DecisionDetail d={sel} />}
      </Modal>
    </div>
  )
}
