import type { Level, EventInfo } from '../types'

export const fmtAmount = (a: number, ccy: string) =>
  `${a.toLocaleString('sr-RS', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} ${ccy}`

// Skor rizika je vjerovatnoća prevare — prikazuje se kao procenat. „1,9%" je
// čitljivije od „0.0185", pogotovo otkad se pragovi izvode iz matrice troška
// i završe na malim vrijednostima.
export const fmtScore = (v: number, digits = 1) =>
  `${(v * 100).toLocaleString('sr-RS', {
    minimumFractionDigits: digits, maximumFractionDigits: digits,
  })}%`

export const fmtTime = (iso: string) => {
  const d = new Date(iso)
  return d.toLocaleString('sr-RS', {
    day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit',
  })
}

export const place = (e: EventInfo) => (e.city ? `${e.city}, ${e.country}` : e.country)

export const fullName = (first?: string | null, last?: string | null) =>
  [first, last].filter(Boolean).join(' ').trim()

export function InfoTip({ title, desc, example }: { title: string; desc: string; example: string }) {
  return (
    <span className="infotip" tabIndex={0} aria-label={`${title}: ${desc} Primjer: ${example}`}>
      <span className="infotip-ic">i</span>
      <span className="infotip-pop" role="tooltip">
        <strong>{title}</strong>
        <span>{desc}</span>
        <em>Primjer: {example}</em>
      </span>
    </span>
  )
}

export const levelClass = (l: Level) =>
  l === 'HIGH' ? 'high' : l === 'MEDIUM' ? 'med' : 'low'

export const levelColor = (l: Level) =>
  l === 'HIGH' ? 'var(--high)' : l === 'MEDIUM' ? 'var(--med)' : 'var(--low)'

// Boja se uvijek izvodi iz nivoa koji je engine dodijelio, nikad iz fiksnih
// granica skora — pragovi se izvode iz matrice troška i mijenjaju se.
// Nivo rizika se korisniku prikazuje na jeziku ostatka interfejsa; API i
// konfiguracija zadržavaju kanonske engleske vrijednosti.
export const LEVEL_LABEL: Record<Level, string> = {
  LOW: 'NIZAK', MEDIUM: 'SREDNJI', HIGH: 'VISOK',
}

export function RiskBadge({ level }: { level: Level }) {
  return <span className={`badge badge--${levelClass(level)}`}>{LEVEL_LABEL[level] ?? level}</span>
}

export function ActionBadge({ action }: { action: string }) {
  const cls = action === 'BLOCK' ? 'high' : action === 'REVIEW' ? 'med'
    : action === 'CHALLENGE' ? 'med' : 'low'
  return <span className={`badge badge--${cls}`}>{action}</span>
}

