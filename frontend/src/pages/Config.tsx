import { useEffect, useState } from 'react'
import type { Rule, Policy, Fusion, CostConfig } from '../types'
import { api } from '../api'
import { InfoTip, fmtScore } from '../components/ui'

type Status = { ok?: string; err?: string }

// Opis šta se unosi u prag + primjer, po feature-u koji se poredi.
const FEATURE_HELP: Record<string, { title: string; desc: string; example: string }> = {
  amount_ratio: {
    title: 'Odnos iznosa',
    desc: 'Koliko puta je iznos transakcije veći od prosječnog iznosa tog korisnika.',
    example: '8 → okida kad je transakcija 8× veća od prosjeka korisnika',
  },
  amount_zscore: {
    title: 'Z-score iznosa',
    desc: 'Za koliko standardnih devijacija iznos odstupa od prosjeka korisnika.',
    example: '4 → okida na iznos 4 devijacije iznad prosjeka (vrlo neobično)',
  },
  txn_count_1h: {
    title: 'Broj transakcija / 1h',
    desc: 'Broj transakcija korisnika u posljednjih 60 minuta.',
    example: '5 → okida kad korisnik ima 5 ili više transakcija za sat',
  },
  txn_count_24h: {
    title: 'Broj transakcija / 24h',
    desc: 'Broj transakcija korisnika u posljednja 24 sata.',
    example: '20 → okida na 20 ili više transakcija dnevno',
  },
  impossible_travel_speed_kmh: {
    title: 'Brzina putovanja (km/h)',
    desc: 'Brzina kretanja između lokacija dvije uzastopne transakcije.',
    example: '1000 → okida iznad 1000 km/h (brže od aviona = fizički nemoguće)',
  },
  is_new_country: {
    title: 'Nova zemlja (0/1)',
    desc: 'Zastavica: je li zemlja transakcije nova za korisnika (1 = da, 0 = ne).',
    example: '1 → okida kada je zemlja prvi put viđena za tog korisnika',
  },
  is_new_device: {
    title: 'Novi uređaj (0/1)',
    desc: 'Zastavica: je li uređaj nov za korisnika (1 = da, 0 = ne).',
    example: '1 → okida kada transakcija dolazi sa novog uređaja',
  },
  is_unusual_hour: {
    title: 'Neuobičajen sat (0/1)',
    desc: 'Zastavica: je li transakcija u neuobičajeno doba (1–5h ujutru).',
    example: '1 → okida za transakcije noću (1–5h)',
  },
}

export default function Config() {
  const [policy, setPolicy] = useState<Policy | null>(null)
  const [fusion, setFusion] = useState<Fusion | null>(null)
  const [rules, setRules] = useState<Rule[]>([])
  const [cost, setCost] = useState<CostConfig | null>(null)
  const [st, setSt] = useState<Record<string, Status>>({})

  const flash = (key: string, s: Status) => {
    setSt((p) => ({ ...p, [key]: s }))
    if (s.ok) setTimeout(() => setSt((p) => ({ ...p, [key]: {} })), 2500)
  }

  useEffect(() => {
    api.policy().then(setPolicy).catch(() => {})
    api.fusion().then(setFusion).catch(() => {})
    api.rules().then(setRules).catch(() => {})
    api.cost().then(setCost).catch(() => {})
  }, [])

  async function savePolicy() {
    if (!policy) return
    try {
      const r = await api.updatePolicy({ mode: policy.mode, low_max: policy.low_max, medium_max: policy.medium_max })
      setPolicy({ ...policy, ...r }); flash('policy', { ok: 'Sačuvano ✓' })
    } catch (e) { flash('policy', { err: String((e as Error).message) }) }
  }
  async function saveFusion() {
    if (!fusion) return
    try { const r = await api.updateFusion(fusion); setFusion(r); flash('fusion', { ok: 'Sačuvano ✓' }) }
    catch (e) { flash('fusion', { err: String((e as Error).message) }) }
  }
  async function saveRules() {
    try {
      const r = await api.updateRules(rules.map((x) => ({
        name: x.name, weight: x.weight, enabled: x.enabled, thresholds: x.thresholds,
      })))
      setRules(r); flash('rules', { ok: 'Sačuvano ✓' })
    } catch (e) { flash('rules', { err: String((e as Error).message) }) }
  }

  async function saveCost(apply: boolean) {
    if (!cost) return
    try {
      const r = await api.updateCost({ ...cost.cost_matrix, ...cost.capacity, apply })
      setCost(r)
      if (r.current) setPolicy((p) => (p ? { ...p, ...r.current } : p))
      flash('cost', {
        ok: apply
          ? (r.applied
            ? `Pragovi primijenjeni: ${fmtScore(r.applied.low_max, 2)} / ${fmtScore(r.applied.medium_max, 2)}`
            : 'Sačuvano, ali nema dovoljno logovanih odluka za izvođenje pragova.')
          : 'Sačuvano ✓',
      })
    } catch (e) { flash('cost', { err: String((e as Error).message) }) }
  }

  const setCostField = (k: keyof CostConfig['cost_matrix'], v: number) =>
    setCost((c) => (c ? { ...c, cost_matrix: { ...c.cost_matrix, [k]: v } } : c))
  const setCapField = (k: keyof CostConfig['capacity'], v: number) =>
    setCost((c) => (c ? { ...c, capacity: { ...c.capacity, [k]: v } } : c))

  // Pragovi se korisniku prikazuju u procentima (skor je vjerovatnoća
  // prevare), a prema API-ju idu kao udio 0..1. Zaokruživanje je samo za
  // prikaz — dok korisnik ne dirne polje, čuva se puna preciznost.
  const asPct = (v: number) => Number((v * 100).toFixed(2))
  const fromPct = (v: number) => v / 100

  const setRule = (name: string, patch: Partial<Rule>) =>
    setRules((rs) => rs.map((r) => (r.name === name ? { ...r, ...patch } : r)))

  const setThreshold = (name: string, i: number, value: number) =>
    setRules((rs) => rs.map((r) => (r.name === name
      ? { ...r, thresholds: r.thresholds.map((t, j) => (j === i ? value : t)) }
      : r)))

  return (
    <div>
      {/* ---------- Režim i pragovi ---------- */}
      <div className="section-title"><h2>Režim i pragovi rizika</h2></div>
      {policy && (
        <div className="card pad">
          <div className="field">
            <label>Režim</label>
            <div className="toggle">
              {(['shadow', 'enforce'] as const).map((m) => (
                <button key={m} className={policy.mode === m ? 'on' : ''}
                  onClick={() => setPolicy({ ...policy, mode: m })}>
                  {m === 'shadow' ? 'SHADOW (loguj)' : 'ENFORCE (izvrši)'}
                </button>
              ))}
            </div>
            <span className="muted" style={{ fontSize: 12 }}>
              {policy.mode === 'shadow'
                ? 'Predložena akcija se loguje, ali se ne izvršava — svaka transakcija prolazi.'
                : 'Predložena akcija se stvarno izvršava.'}
            </span>
          </div>
          <div className="field">
            <label>NIZAK ako je rizik ≤</label>
            <input type="number" step="0.1" min="0" max="100" value={asPct(policy.low_max)}
              onChange={(e) => setPolicy({ ...policy, low_max: fromPct(Number(e.target.value)) })} />
            <span className="muted" style={{ fontSize: 12 }}>%</span>
          </div>
          <div className="field">
            <label>SREDNJI ako je rizik ≤</label>
            <input type="number" step="0.1" min="0" max="100" value={asPct(policy.medium_max)}
              onChange={(e) => setPolicy({ ...policy, medium_max: fromPct(Number(e.target.value)) })} />
            <span className="muted" style={{ fontSize: 12 }}>% · iznad toga = VISOK</span>
          </div>
          <div className="field">
            <label>Nivo → akcija</label>
            <span className="mono" style={{ fontSize: 12.5 }}>
              {Object.entries(policy.level_actions).map(([k, v]) => `${k}→${v}`).join('  ·  ')}
            </span>
          </div>
          <div className="saverow">
            <button className="btn btn--brand" onClick={savePolicy}>Sačuvaj</button>
            {st.policy?.ok && <span className="saved">{st.policy.ok}</span>}
            {st.policy?.err && <span className="saveerr">{st.policy.err}</span>}
          </div>
        </div>
      )}


      {/* ---------- Ekonomija odluke ---------- */}
      <div className="section-title"><h2>Ekonomija odluke</h2></div>
      {cost && (
        <div className="card pad">
          <p className="muted" style={{ fontSize: 12.5, marginTop: 0 }}>
            Pragovi rizika ne moraju biti pogođeni „od oka". Ako je skor kalibrisana
            vjerovatnoća prevare, optimalan prag slijedi iz odnosa troškova: koliko
            košta propuštena prevara u odnosu na trenje koje pravimo korisniku.
          </p>
          <div className="field">
            <label>Gubitak po prevari (EUR)</label>
            <input type="number" step="10" min="0" value={cost.cost_matrix.avg_fraud_loss}
              onChange={(e) => setCostField('avg_fraud_loss', Number(e.target.value))} />
            <InfoTip title="Prosječan gubitak" desc="Koliko banka izgubi kad prevara prođe."
              example="250 → prosječna neovlašćena transakcija košta 250 EUR" />
          </div>
          <div className="field">
            <label>Trošak izazova (EUR)</label>
            <input type="number" step="0.1" min="0" value={cost.cost_matrix.challenge_cost}
              onChange={(e) => setCostField('challenge_cost', Number(e.target.value))} />
            <InfoTip title="Trošak 2FA izazova" desc="Trenje koje pravimo korisniku (SMS, prekid kupovine)."
              example="0.5 → svaki izazov košta pola eura" />
          </div>
          <div className="field">
            <label>Trošak pregleda (EUR)</label>
            <input type="number" step="0.5" min="0" value={cost.cost_matrix.review_cost}
              onChange={(e) => setCostField('review_cost', Number(e.target.value))} />
            <InfoTip title="Trošak ručnog pregleda" desc="Vrijeme analitičara po slučaju."
              example="8 → jedan pregled košta 8 EUR" />
          </div>
          <div className="field">
            <label>Izazov zaustavlja</label>
            <input type="number" step="1" min="0" max="100" value={asPct(cost.cost_matrix.challenge_stop_rate)}
              onChange={(e) => setCostField('challenge_stop_rate', fromPct(Number(e.target.value)))} />
            <span className="muted" style={{ fontSize: 12 }}>% prevara koje 2FA spriječi</span>
          </div>
          <div className="field">
            <label>Pregled zaustavlja</label>
            <input type="number" step="1" min="0" max="100" value={asPct(cost.cost_matrix.review_stop_rate)}
              onChange={(e) => setCostField('review_stop_rate', fromPct(Number(e.target.value)))} />
            <span className="muted" style={{ fontSize: 12 }}>% prevara koje ručni pregled spriječi</span>
          </div>

          <div className="field">
            <label>Najviše izazova</label>
            <input type="number" step="0.5" min="0" max="100" value={asPct(cost.capacity.max_challenge_rate)}
              onChange={(e) => setCapField('max_challenge_rate', fromPct(Number(e.target.value)))} />
            <span className="muted" style={{ fontSize: 12 }}>%</span>
            <InfoTip title="Budžet trenja"
              desc="Koliki udio transakcija smije dobiti 2FA. Bez ovog ograničenja optimizacija traži izazov na skoro svaku transakciju."
              example="5% → najviše svaka dvadeseta transakcija" />
          </div>
          <div className="field">
            <label>Najviše pregleda</label>
            <input type="number" step="0.1" min="0" max="100" value={asPct(cost.capacity.max_review_rate)}
              onChange={(e) => setCapField('max_review_rate', fromPct(Number(e.target.value)))} />
            <span className="muted" style={{ fontSize: 12 }}>%</span>
            <InfoTip title="Kapacitet analitičara" desc="Koliko slučajeva tim stvarno može pregledati."
              example="1% → najviše svaka stota transakcija" />
          </div>

          <div className="field">
            <label>Izvedeni pragovi</label>
            <span className="mono" style={{ fontSize: 12.5 }}>
              zatvorena forma: {fmtScore(cost.closed_form.low_max, 2)} / {fmtScore(cost.closed_form.medium_max, 2)}
              {'  ·  '}trenutno u primjeni: {fmtScore(cost.current.low_max, 2)} / {fmtScore(cost.current.medium_max, 2)}
            </span>
          </div>

          <div className="saverow">
            <button className="btn" onClick={() => saveCost(false)}>Sačuvaj</button>
            <button className="btn btn--brand" onClick={() => saveCost(true)}>
              Izvedi i primijeni pragove
            </button>
            {st.cost?.ok && <span className="saved">{st.cost.ok}</span>}
            {st.cost?.err && <span className="saveerr">{st.cost.err}</span>}
          </div>
        </div>
      )}

      {/* ---------- Fusion ---------- */}
      <div className="section-title"><h2>Fusion (spajanje pravila i ML)</h2></div>
      {fusion && (
        <div className="card pad">
          <div className="field">
            <label>Strategija</label>
            <select value={fusion.strategy} onChange={(e) => setFusion({ ...fusion, strategy: e.target.value })}>
              <option value="weighted">weighted</option>
              <option value="cascade">cascade</option>
              <option value="stacking">stacking</option>
            </select>
          </div>
          <div className="field">
            <label>Težina pravila</label>
            <input type="number" step="0.05" min="0" value={fusion.rules_weight}
              onChange={(e) => setFusion({ ...fusion, rules_weight: Number(e.target.value) })} />
          </div>
          <div className="field">
            <label>Težina ML</label>
            <input type="number" step="0.05" min="0" value={fusion.ml_weight}
              onChange={(e) => setFusion({ ...fusion, ml_weight: Number(e.target.value) })} />
          </div>
          <div className="field">
            <label>Cascade prag</label>
            <input type="number" step="1" min="0" max="100" value={asPct(fusion.cascade_threshold)}
              onChange={(e) => setFusion({ ...fusion, cascade_threshold: fromPct(Number(e.target.value)) })} />
            <span className="muted" style={{ fontSize: 12 }}>%</span>
          </div>
          <div className="saverow">
            <button className="btn btn--brand" onClick={saveFusion}>Sačuvaj</button>
            {st.fusion?.ok && <span className="saved">{st.fusion.ok}</span>}
            {st.fusion?.err && <span className="saveerr">{st.fusion.err}</span>}
          </div>
        </div>
      )}

      {/* ---------- Pravila ---------- */}
      <div className="section-title"><h2>Poslovna pravila (banking pack)</h2>
        <span className="muted" style={{ fontSize: 12 }}>{rules.length} pravila</span></div>
      <div className="card">
        <table>
          <thead>
            <tr><th>Pravilo</th><th>Uslov</th><th className="num">Težina</th><th>Aktivno</th><th>Objašnjenje</th></tr>
          </thead>
          <tbody>
            {rules.map((r) => (
              <tr key={r.name} style={{ opacity: r.enabled ? 1 : 0.5 }}>
                <td className="mono">{r.name}</td>
                <td className="mono muted">
                  {r.condition_parts.map((part, i) => (
                    <span key={i}>
                      {part}
                      {i < r.thresholds.length && (() => {
                        const help = FEATURE_HELP[r.threshold_features[i]]
                        return (
                          <>
                            <input type="number" className="thr" value={r.thresholds[i]}
                              onChange={(e) => setThreshold(r.name, i, Number(e.target.value))} />
                            {help && <InfoTip title={help.title} desc={help.desc} example={help.example} />}
                          </>
                        )
                      })()}
                    </span>
                  ))}
                </td>
                <td>
                  <input type="number" step="0.05" min="0" max="1" value={r.weight}
                    style={{ width: 70 }}
                    onChange={(e) => setRule(r.name, { weight: Number(e.target.value) })} />
                </td>
                <td>
                  <input type="checkbox" checked={r.enabled}
                    onChange={(e) => setRule(r.name, { enabled: e.target.checked })} />
                </td>
                <td>{r.explanation}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="pad saverow">
          <button className="btn btn--brand" onClick={saveRules}>Sačuvaj pravila</button>
          {st.rules?.ok && <span className="saved">{st.rules.ok}</span>}
          {st.rules?.err && <span className="saveerr">{st.rules.err}</span>}
        </div>
      </div>
    </div>
  )
}
