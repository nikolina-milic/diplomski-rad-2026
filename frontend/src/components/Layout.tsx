import { NavLink, Outlet, useLocation } from 'react-router-dom'

const NAV = [
  { to: '/', label: 'Live monitor', ic: '●', end: true },
  { to: '/dogadjaji', label: 'Svi događaji', ic: '≡' },
  { to: '/review', label: 'Review queue', ic: '⚑' },
  { to: '/metrics', label: 'Metrike', ic: '▤' },
  { to: '/config', label: 'Konfiguracija', ic: '⚙' },
  { to: '/registry', label: 'Model registry', ic: '⬡' },
]

const TITLES: Record<string, { h: string; s: string }> = {
  '/': { h: 'Live monitor', s: 'Tok odluka o riziku u realnom vremenu' },
  '/dogadjaji': { h: 'Svi događaji', s: 'Kompletan audit log odluka, sa filterom po nivou rizika' },
  '/review': { h: 'Review queue', s: 'Slučajevi za ručni pregled analitičara' },
  '/metrics': { h: 'Metrike i evaluacija', s: 'Performanse modela i konfuziona matrica' },
  '/config': { h: 'Konfiguracija (banking pack)', s: 'Pravila, politike i režim odlučivanja' },
  '/registry': { h: 'Model registry', s: 'Verzije modela, champion / challenger, promocija' },
}

export default function Layout() {
  const { pathname } = useLocation()
  const t = TITLES[pathname] ?? TITLES['/']
  return (
    <div className="app">
      <aside className="rail">
        <div className="brand">
          <span className="dot" />
          <span>SPR<small>SISTEM ZA PROCJENU RIZIKA</small></span>
        </div>
        <nav className="nav">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.end}
              className={({ isActive }) => (isActive ? 'active' : '')}>
              <span className="ic">{n.ic}</span>
              <span className="label">{n.label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="foot">Hibridni engine · v0.1<br />pravila + ML + fusion</div>
      </aside>
      <main className="main">
        <div className="topbar">
          <h1>{t.h}</h1>
          <span className="sub">{t.s}</span>
        </div>
        <div className="content"><Outlet /></div>
      </main>
    </div>
  )
}
