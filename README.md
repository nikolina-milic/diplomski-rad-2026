# SPR — Sistem za procjenu rizika

Hibridni sistem za upravljanje rizikom digitalnih događaja u realnom vremenu. Na osnovu ponašanja
korisnika i kontekstualnih faktora (vrijeme, lokacija, uređaj) i **hibridnog modela odlučivanja**
(poslovna pravila + mašinsko učenje + kalibrisani fusion sloj) procjenjuje vjerovatnoću prevare i
predlaže ili izvršava odgovarajuću akciju (`ALLOW` / `CHALLENGE` / `REVIEW` / `BLOCK`).

Domenski-nezavisno jezgro sa „domain pack" mehanizmom; u ovoj verziji je implementiran **bankarski
domen** (transakcije, kartice) na **sintetičkim podacima**. Diplomski rad — kompletna, funkcionalna
aplikacija sa backend API-jem, interaktivnim dashboardom i mjerljivom evaluacijom.

---

## Brzi start

```bash
# Terminal 1 — backend
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/uvicorn spr.api.main:app --port 8000

# Terminal 2 — frontend
cd frontend && npm install && npm run dev
```

Otvori **<http://localhost:5173>**. (API i Swagger: <http://localhost:8000/docs>.)

> Ako je port 8000 zauzet, pokreni backend na drugom portu i proslijedi ga frontendu:
> `.venv/bin/uvicorn spr.api.main:app --port 8001` i
> `VITE_API_BASE=http://localhost:8001 npm run dev`.

---

## Arhitektura

```
 Sintetički generator ──event──▶ BACKEND (FastAPI)
 (batch + live stream)          │
                                │  DECISION ENGINE
                                │  context → rules → ML(SHAP) → fusion → policy → guardrails
                                │        │           kalibracija + stacking   pragovi iz troška
                                │        │
                                │  Audit log + Feedback (SQLite/Postgres)
                                │  Learning: retrening → champion/challenger → promocija
                                │  Monitoring: drift (PSI/KS)
                                └── REST /  WebSocket
                                        │
                                FRONTEND (React + TS)
                                Live monitor · Review · Metrike · Config · Registry
```

Ključni moduli (`src/spr/`): `generator/`, `context/`, `rules/`, `ml/`, `fusion/`, `policy/`,
`engine/`, `learning/`, `persistence/`, `api/`, `evaluation/`, `domain/` (banking pack).

---

## Kako sistem donosi odluku

1. **Kontekst** — iz istorije korisnika (stanje *prije* događaja, bez curenja podataka) računa se
   12 obilježja: odnos i z-score iznosa, učestalost, nov uređaj/zemlja, doba dana, brzina kretanja
   između lokacija, udaljenost od doma.
2. **Pravila** — deterministički izrazi nad obilježjima; skor je noisy-OR kombinacija težina
   okinutih pravila: `1 − Π(1 − wᵢ)`.
3. **ML** — RandomForest sa `class_weight="balanced"`, uz SHAP doprinose po obilježju.
4. **Fusion** — oba skora se **kalibrišu** u vjerovatnoće (izotonijska regresija), pa ih meta-model
   (logistička regresija nad `[r, m, r·m]`) kombinuje. Kalibratori i meta-model uče se nad
   **out-of-fold** predikcijama, jer su skorovi baznog modela na sopstvenom trening skupu
   preoptimistični.
5. **Politika** — pragovi nivoa rizika **izvedeni iz matrice troška** (koliko košta propuštena
   prevara u odnosu na trenje korisnika i vrijeme analitičara), uz ograničenja kapaciteta.
6. **Guardrails** — tvrda ograničenja koja nadjačavaju politiku (npr. regulatorni prag iznosa).
7. **Režim** — u `shadow` režimu predložena akcija se **loguje ali ne izvršava**
   (`executed_action = ALLOW`); u `enforce` se izvršava.

---

## Preduslovi

- **Python 3.12+**
- **Node.js 20+** i **npm** (za frontend)

---

## Instalacija

### Backend

```bash
cd SPR
python3.12 -m venv .venv                 # ako venv već ne postoji
.venv/bin/pip install -e ".[dev]"        # instalira paket + sve zavisnosti + pytest
```

### Frontend

```bash
cd frontend
npm install
```

---

## Pokretanje

Potrebna su **dva terminala** (backend + frontend).

### 1) Backend (API)

```bash
cd SPR
.venv/bin/uvicorn spr.api.main:app --port 8000
```

> Pri pokretanju backend istrenira početni model (`v1`), izračuna out-of-fold skorove, kalibriše
> fusion sloj i izvede pragove iz matrice troška — traje petnaestak sekundi. Kad ispiše da je
> spreman, API je dostupan. Model `v1` se odmah upisuje u registry kao champion.

- API: <http://localhost:8000>
- Interaktivna dokumentacija (Swagger): <http://localhost:8000/docs>

### 2) Frontend (dashboard)

```bash
cd frontend
npm run dev
```

- Dashboard: <http://localhost:5173>

Otvori dashboard u browseru. Nije potrebna prijava. Live monitor se automatski povezuje na
backend i počinje da prikazuje tok transakcija.

---

## Dashboard — ekrani

| Ekran | Opis |
|-------|------|
| **Live monitor** | Tri kolone po nivou rizika (nizak / srednji / visok), po 20 najnovijih odluka u svakoj. Klik na karticu otvara modal sa punim objašnjenjem (pravila + SHAP + skorovi + predložena vs. izvršena akcija). Iznad kolona živi brojači po akciji; u zaglavlju svake kolone ukupan broj i dugme **Pogledaj sve**. |
| **Svi događaji** | Kompletan audit log u tabeli, sa filterom po nivou rizika (filter stoji u URL-u, pa se pogled može podijeliti linkom) i izborom koliko najnovijih učitati. Klik na red otvara isti modal. |
| **Review queue** | Slučajevi označeni `REVIEW`. Dugmad **Prevara / Legitimno** šalju feedback u petlju učenja. |
| **Metrike** | Konfuziona matrica, PR-AUC/F1/preciznost/odziv, poređenje pravila vs ML vs hibrid, PR kriva, **kvalitet kalibracije (Brier, ECE, reliability dijagram)**, **trošak politike**, **drift (PSI/KS)**, latencija. |
| **Konfiguracija** | Banking pack: pravila, pragovi, nivo→akcija mapiranje, shadow/enforce režim, **ekonomija odluke** (matrica troška + kapacitet → izvedeni pragovi). |
| **Model registry** | Verzije modela (champion/challenger/retired) + metrike. Dugme **Pokreni retrening** trenira challenger, poredi ga sa championom i promoviše ako je bolji. |

---

## Testovi

```bash
cd SPR
.venv/bin/python -m pytest -q
```

184 testa; pokrivaju generator (uključujući **provjere realizma** — da klase nisu trivijalno
separabilne), context, pravila, ML, kalibraciju, fusion model, trošak/pragove, politiku i režim,
engine, API, perzistenciju, učenje, drift i evaluaciju.

---

## Evaluacija (za pisani rad)

```bash
.venv/bin/python -m spr.evaluation                    # brzo, 1 seed
.venv/bin/python -m spr.evaluation --seeds 10         # + značajnost razlika
.venv/bin/python -m spr.evaluation --seeds 10 --full  # + svi eksperimenti (sporo)
```

Izvještaj se zapisuje u `reports/evaluation.json`. Sa `--full` uključuje:

| Eksperiment | Pitanje na koje odgovara |
|-------------|--------------------------|
| Više seedova + Wilcoxon | Je li razlika između pristupa stvarna ili šum? |
| Kalibracija (Brier, ECE) | Znači li skor 0.8 zaista „80% šanse za prevaru"? |
| Trošak politike | Koliko nas koštaju proizvoljno izabrani pragovi? |
| `model_zoo` | Zašto baš RandomForest, a ne logistička regresija ili boosting? |
| `rule_ablation` | Koje pravilo stvarno nosi odluku? |
| `feature_ablation` | Koje obilježje stvarno nosi odluku? |
| `novel_attack` | Šta se dešava sa obrascem napada kojeg nema u trening podacima? |
| `label_scarcity` | Koliko labela treba da ML prestigne pravila? |
| `feedback_value` | Koliko stvarno doprinosi feedback analitičara? |
| `drift_scenarios` | Reaguje li detektor drifta na poznato pomjeranje? |

> Napomena o broju seedova: Wilcoxonov test sa `n` seedova ne može dati dvostranu p-vrijednost
> manju od `2/2ⁿ`. Sa 5 seedova minimum je 0.0625, pa nijedna razlika ne može ispasti značajna —
> koristi bar `--seeds 6`, praktično 10.

---

## Pregled API ruta

| Metoda | Ruta | Opis |
|--------|------|------|
| GET | `/health` | Status, verzija modela, režim |
| POST | `/score` | Skoruje jedan događaj → puna odluka + objašnjenje |
| GET | `/rules` | Aktivna poslovna pravila |
| GET | `/policy` | Pragovi, nivo→akcija, režim |
| GET | `/fusion` | Konfiguracija fusion sloja |
| GET | `/cost` | Matrica troška, kapacitet i pragovi koji iz njih slijede |
| PUT | `/config/cost` | Izmjena ekonomskih parametara; `apply=true` izvodi i primjenjuje pragove |
| GET | `/calibration` | Stanje kalibracije + naučene težine meta-modela |
| PUT | `/config/policy` · `/config/fusion` · `/config/rules` | Izmjena konfiguracije |
| WS | `/ws/stream?interval=5` | Live tok skorovanih događaja (interval u sekundama) |
| GET | `/decisions?limit=50&level=HIGH` | Poslednje logovane odluke, opciono filtrirane po nivou rizika |
| GET | `/decisions/count` | Ukupan broj odluka + raspodjela po nivou i po akciji (SQL prebrojavanje) |
| GET | `/review-queue` | Slučajevi za ručni pregled |
| POST | `/feedback` | Analitičarski ishod (`{decision_id, label, source}`) |
| GET | `/models` | Registry verzija modela |
| POST | `/retrain` | Retrening → champion/challenger → promocija (verziju bira backend) |
| GET | `/stats?window=200` | Brojači akcija/nivoa u posljednjih N odluka (dashboard koristi `/decisions/count`, koji broji sve) |
| GET | `/metrics` | Konfuziona matrica + metrike iz logovanih odluka |
| GET | `/drift?window=500` | Pomjeranje raspodjele nedavnih odluka u odnosu na ranije |
| GET | `/evaluation?n_test=800` | Poređenje pristupa + PR krive + kalibracija + trošak + latencija |

---

## Konfiguracija i napomene

- **Sintetički podaci.** Nema stvarnih transakcija. Generator (`src/spr/generator/`) simulira
  persone sa **kontinuitetom stanja**: korisnici putuju (uvjerljivom brzinom), mijenjaju uređaje,
  povremeno naprave veliku kupovinu i troše van uobičajenih sati. Prevare (account takeover, card
  testing, impossible travel, sudden large amount, new country/device) ubacuju se u tu istoriju i
  **namjerno dijele prostor obilježja sa legitimnim ponašanjem**. Bez toga bi „nova zemlja"
  implicirala prevaru, klasifikacija bi bila trivijalna, a evaluacija besmislena.
- **Udio prevara.** Podrazumijevano 2% — red veličine iznad stvarnog (0.1–1%), ali dovoljno da
  metrike na test setu budu stabilne. Menja se preko `--fraud-rate`.
- **Ground truth ne dolazi od klijenta.** `POST /score` **ignoriše** `is_fraud` iz tijela zahtjeva:
  u stvarnom sistemu ishod nije poznat u trenutku odluke. Labela stiže kasnije, kroz `/feedback`.
  Za demo i testove postoji `POST /score?record_truth=true`.
- **Režim `shadow`.** Sistem loguje predložene akcije, ne izvršava ih — vidljivo kroz
  `executed_action`. Promjena praga/režima je u `src/spr/domain/banking_pack.py`
  (`BANKING_POLICY`, `BANKING_FUSION`, `BANKING_RULES`, `BANKING_GUARDRAILS`, `BANKING_COST`,
  `BANKING_CAPACITY`) ili uživo preko `/config/*`.
- **Baza je trajna.** Podrazumijevano SQLite fajl `spr.db` u korijenu; podaci ostaju između
  pokretanja, a nedostajuće kolone se dodaju pri startu (`migrate_sqlite`). Za čist početak ručno
  obriši `spr.db`. Za PostgreSQL postavi `DATABASE_URL` i proslijedi ga u `make_session_factory`
  (`src/spr/api/main.py`).
- **Bez autentifikacije.** Dashboard je otvoren — auth/RBAC je van opsega ove verzije.
- **CORS** je otvoren radi lokalnog razvoja (frontend na `:5173` → backend na `:8000`).

---

## Struktura projekta

```
SPR/
├── src/spr/              # backend paket
│   ├── generator/        # sintetički generator (persone, putovanja, obrasci prevare)
│   ├── context/          # feature engineering + kontekst store
│   ├── rules/            # deterministički rules evaluator
│   ├── ml/               # model zoo (RF/logistic/boosting/isolation) + SHAP + kalibracija
│   ├── fusion/           # strategije, guardrails, kalibrisani fusion model (stacking)
│   ├── policy/           # nivo → akcija, shadow/enforce, pragovi iz matrice troška
│   ├── engine/           # orkestracija odluke + factory
│   ├── learning/         # metrike, retrening, champion/challenger, drift
│   ├── persistence/      # SQLAlchemy modeli + repozitorijum + migracija
│   ├── evaluation/       # izvještaj, eksperimenti, CLI (python -m spr.evaluation)
│   ├── api/              # FastAPI aplikacija (app.py, main.py)
│   └── domain/           # šema događaja + banking domain pack
├── frontend/             # React + TypeScript dashboard (Vite)
├── tests/                # pytest testovi
├── reports/              # izvještaji evaluacije (generisano)
├── pyproject.toml
└── README.md
```
