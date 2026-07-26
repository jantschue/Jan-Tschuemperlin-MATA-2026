import { useMemo, useState, useEffect } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Cell
} from 'recharts'
import { useModelWeights } from '../hooks/useModelWeights.js'
import { mlpForward, linearForward } from '../utils/modelForward.js'
import {
  buildFeatureVector,
  DEFAULT_INPUTS,
  MONTHS,
  WEEKDAYS,
  WEATHER_OPTIONS
} from '../utils/featureBuilder.js'

// Ostersonntag nach Butcher's Algorithm
function easterSunday(year) {
  const a = year % 19
  const b = Math.floor(year / 100)
  const c = year % 100
  const d = Math.floor(b / 4)
  const e = b % 4
  const f = Math.floor((b + 8) / 25)
  const g = Math.floor((b - f + 1) / 3)
  const h = (19 * a + b - d - g + 15) % 30
  const i = Math.floor(c / 4)
  const k = c % 4
  const l = (32 + 2 * e + 2 * i - h - k) % 7
  const m = Math.floor((a + 11 * h + 22 * l) / 451)
  const month = Math.floor((h + l - 7 * m + 114) / 31)
  const day = ((h + l - 7 * m + 114) % 31) + 1
  return new Date(year, month - 1, day)
}

// Prüft ob ein Datum ein Schweizer Feiertag ist
function isSwissHoliday(date) {
  const y = date.getFullYear()
  const m = date.getMonth() + 1
  const d = date.getDate()
  if ((m === 1 && d === 1) || (m === 8 && d === 1) ||
      (m === 12 && d === 25) || (m === 12 && d === 26)) return true
  const easter = easterSunday(y)
  const diff = Math.round((new Date(y, m - 1, d) - easter) / 86400000)
  return diff === -2 || diff === 1 || diff === 39 || diff === 50
}

// WMO-Wettercodes → weather_cat (0=Clear, 1=Cloudy, 3=Rain, 4=Snow)
function wmoCat(code) {
  if (code === 0 || code === 1) return 0
  if (code <= 3 || code === 45 || code === 48) return 1
  if ((code >= 71 && code <= 77) || code === 85 || code === 86) return 4
  return 3
}

// Heutige Datums-Eingaben (ohne Wetter – wird separat geladen)
function todayInputs() {
  const now = new Date()
  const jsDay = now.getDay()
  return {
    ...DEFAULT_INPUTS,
    year: now.getFullYear(),
    hour: now.getHours(),
    dayOfWeek: jsDay === 0 ? 6 : jsDay - 1,
    month: now.getMonth() + 1,
    day: now.getDate(),
    isHoliday: isSwissHoliday(now)
  }
}

// Animations-Wrapper: rendert die Zahl bei Wert-Wechsel mit fade+slide-in
function AnimatedNumber({ value }) {
  return (
    <div
      key={value}
      className="animate-prediction metric-display text-[3.4rem] sm:text-[3.8rem] leading-[0.95]"
    >
      {value}
    </div>
  )
}

function PredictionCard({ title, value, r2, color, accent }) {
  return (
    <div className="card p-6 flex flex-col relative overflow-hidden">
      <div
        aria-hidden
        className="absolute inset-x-0 top-0 h-px"
        style={{ background: color, opacity: 0.5 }}
      />
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-2.5">
          <span
            className="inline-block w-1.5 h-1.5 rounded-full"
            style={{ background: color }}
          />
          <h3 className="text-[0.95rem] tracking-tightish">{title}</h3>
          {accent && (
            <span className="text-[10px] eyebrow text-[var(--accent)]">primär</span>
          )}
        </div>
        <span
          className="text-[11px] font-mono px-2 py-0.5 rounded-full text-[var(--text-secondary)]"
          style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)' }}
        >
          R² <span style={{ color }}>{r2.toFixed(2)}</span>
        </span>
      </div>
      <AnimatedNumber value={value} />
      <div className="text-xs text-[var(--text-muted)] mt-3">
        Fahrzeuge pro Stunde
      </div>
    </div>
  )
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload || !payload.length) return null
  return (
    <div className="chart-tooltip">
      <div className="font-mono mb-1">{String(label).padStart(2, '0')}:00 Uhr</div>
      {payload.map((p) => (
        <div key={p.dataKey} className="row">
          <span className="label" style={{ color: p.color }}>
            {p.name}
          </span>
          <span>{p.value}</span>
        </div>
      ))}
    </div>
  )
}

export default function LiveVorhersage({ station, onGoToMap }) {
  const { loading, weights, error } = useModelWeights(station.id)
  const [inputs, setInputs] = useState(todayInputs)

  // Aktuelles Wetter für Kanton Schwyz (47.02°N, 8.65°E) von Open-Meteo laden
  useEffect(() => {
    fetch(
      'https://api.open-meteo.com/v1/forecast?latitude=47.02&longitude=8.65' +
      '&current=temperature_2m,precipitation,snowfall,weathercode,cloud_cover&timezone=Europe%2FZurich'
    )
      .then(r => r.json())
      .then(data => {
        const cur = data.current
        if (!cur) return
        const sun = Math.max(0, 1 - (cur.cloud_cover ?? 50) / 100)
        setInputs(s => ({
          ...s,
          temp: Math.round(cur.temperature_2m ?? s.temp),
          rain: Math.min(20, parseFloat((cur.precipitation ?? 0).toFixed(1))),
          snow: Math.min(10, parseFloat(((cur.snowfall ?? 0) * 10).toFixed(1))),
          sun: parseFloat(sun.toFixed(2)),
          weatherCat: wmoCat(cur.weathercode ?? cur.weather_code ?? 0)
        }))
      })
      .catch(() => {})
  }, [])

  // Festes Y-Maximum über alle 24 Stunden mit DEFAULT_INPUTS – unabhängig von Eingaben
  const fixedYMax = useMemo(() => {
    if (!weights) return 2000
    const maxVal = Math.max(...Array.from({ length: 24 }, (_, h) => {
      const v = buildFeatureVector({ ...DEFAULT_INPUTS, hour: h }, weights.mlp.features)
      return Math.max(mlpForward(v, weights.mlp), linearForward(v, weights.linear))
    }))
    return Math.ceil(maxVal / 100) * 100
  }, [weights])

  const update = (key) => (e) => {
    const val = e.target ? e.target.value : e
    setInputs((s) => ({ ...s, [key]: typeof val === 'string' ? Number(val) : val }))
  }

  // Vorhersage für aktuelle Eingaben
  const prediction = useMemo(() => {
    if (!weights) return { mlp: 0, linear: 0 }
    const v = buildFeatureVector(inputs, weights.mlp.features)
    return {
      mlp: mlpForward(v, weights.mlp),
      linear: linearForward(v, weights.linear)
    }
  }, [inputs, weights])

  // 24h-Verlauf für das Balkendiagramm
  const dayPredictions = useMemo(() => {
    if (!weights) return []
    return Array.from({ length: 24 }, (_, h) => {
      const v = buildFeatureVector({ ...inputs, hour: h }, weights.mlp.features)
      return {
        hour: h,
        MLP: mlpForward(v, weights.mlp),
        LR: linearForward(v, weights.linear)
      }
    })
  }, [inputs, weights])

  if (loading) {
    return (
      <div className="card p-16 text-center">
        <div className="inline-flex items-center gap-3 text-[var(--text-muted)] text-sm">
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-[var(--accent)] animate-pulse" />
          Modell wird geladen
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="card p-10 max-w-md mx-auto text-center">
        <div className="eyebrow mb-2">Fehler</div>
        <h3 className="mb-2">Modell nicht verfügbar</h3>
        <p className="text-sm text-[var(--text-muted)]">
          Gewichte für Station <span className="font-mono">{station.id}</span> noch nicht
          exportiert.
        </p>
      </div>
    )
  }

  const mlpR2 = station.metrics.mlp.r2
  const lrR2 = station.metrics.linear.r2

  return (
    <div className="space-y-8">
      <div className="stagger-in">
        <h1 className="mb-2">
          <span className="text-[var(--text-secondary)]">Was sagt das Modell</span>{' '}
          – für diese Bedingungen?
        </h1>
        <p className="text-sm text-[var(--text-secondary)] max-w-prose">
          Stelle Uhrzeit, Wetter und Kalender frei ein. MLP und lineare Regression
          rechnen die Prognose live im Browser.
        </p>
      </div>

      <div className="grid lg:grid-cols-5 gap-6">
        {/* Linkes Panel: Eingabeformular */}
        <aside className="lg:col-span-2 card p-6 stagger-in">
          <div className="flex items-baseline justify-between mb-5">
            <h2>Eingaben</h2>
            <span className="eyebrow">Parameter</span>
          </div>

          <div className="mb-6 pb-5 border-b border-[var(--border)]">
            <div className="eyebrow mb-1.5">Station</div>
            <div className="flex items-center justify-between gap-3">
              <div className="text-sm text-[var(--text-primary)] truncate">
                {station.name} <span className="text-[var(--text-muted)]">· {station.direction}</span>
              </div>
              <button onClick={onGoToMap} className="btn-link shrink-0">
                Ändern
              </button>
            </div>
          </div>

        <div className="space-y-5">
          <Slider
            label="Uhrzeit"
            value={inputs.hour}
            min={0}
            max={23}
            onChange={update('hour')}
            display={`${String(inputs.hour).padStart(2, '0')}:00 Uhr`}
          />

          <Field label="Wochentag">
            <select value={inputs.dayOfWeek} onChange={update('dayOfWeek')}>
              {WEEKDAYS.map((d, i) => (
                <option key={i} value={i}>
                  {d}
                </option>
              ))}
            </select>
          </Field>

          <Field label="Monat">
            <select value={inputs.month} onChange={update('month')}>
              {MONTHS.map((m, i) => (
                <option key={i} value={i + 1}>
                  {m}
                </option>
              ))}
            </select>
          </Field>

          <Slider
            label="Temperatur"
            value={inputs.temp}
            min={-10}
            max={35}
            step={1}
            onChange={update('temp')}
            display={`${inputs.temp}°C`}
          />

          <Slider
            label="Niederschlag (rain_1h)"
            value={inputs.rain}
            min={0}
            max={20}
            step={0.5}
            onChange={update('rain')}
            display={`${inputs.rain.toFixed(1)} mm`}
          />

          <Slider
            label="Sonnenstunden (sun_1h)"
            value={inputs.sun}
            min={0}
            max={1}
            step={0.05}
            onChange={update('sun')}
            display={inputs.sun.toFixed(2)}
          />

          <Slider
            label="Schnee (snow_1h)"
            value={inputs.snow}
            min={0}
            max={10}
            step={0.5}
            onChange={update('snow')}
            display={`${inputs.snow.toFixed(1)} mm`}
          />

          <Field label="Wetterkategorie">
            <div className="radio-pill">
              {WEATHER_OPTIONS.map((w) => (
                <span key={w.id}>
                  <input
                    type="radio"
                    name="weather"
                    id={`w-${w.id}`}
                    checked={inputs.weatherCat === w.cat}
                    onChange={() => setInputs((s) => ({ ...s, weatherCat: w.cat }))}
                  />
                  <label htmlFor={`w-${w.id}`}>{w.label}</label>
                </span>
              ))}
            </div>
          </Field>

          <Field label="Feiertag">
            <label className="toggle">
              <input
                type="checkbox"
                checked={inputs.isHoliday}
                onChange={(e) =>
                  setInputs((s) => ({ ...s, isHoliday: e.target.checked }))
                }
              />
              <span className="slider" />
            </label>
          </Field>

          <Field label="Schulferien">
            <label className="toggle">
              <input
                type="checkbox"
                checked={inputs.isSchoolHoliday}
                onChange={(e) =>
                  setInputs((s) => ({ ...s, isSchoolHoliday: e.target.checked }))
                }
              />
              <span className="slider" />
            </label>
          </Field>
        </div>
      </aside>

        {/* Rechtes Panel: Ergebnisse */}
        <section className="lg:col-span-3 space-y-6 stagger-in" style={{ animationDelay: '80ms' }}>
          <div className="grid sm:grid-cols-2 gap-4">
            <PredictionCard
              title="MLP"
              value={prediction.mlp}
              r2={mlpR2}
              color="var(--accent)"
              accent
            />
            <PredictionCard
              title="Lineare Regression"
              value={prediction.linear}
              r2={lrR2}
              color="var(--lr-color)"
            />
          </div>

          <div className="card p-6">
            <div className="flex items-baseline justify-between mb-4">
              <h3>24-Stunden Verlauf</h3>
              <span className="text-xs text-[var(--text-muted)]">
                Aktuelle Stunde hervorgehoben
              </span>
            </div>
            <div style={{ width: '100%', height: 280 }}>
              <ResponsiveContainer>
                <BarChart data={dayPredictions} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
                  <CartesianGrid stroke="var(--border)" strokeDasharray="2 3" />
                  <XAxis
                    dataKey="hour"
                    stroke="var(--text-muted)"
                    tick={{ fontSize: 11 }}
                  />
                  <YAxis domain={[0, fixedYMax]} stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
                  <Tooltip
                    cursor={{ fill: 'rgba(255,255,255,0.03)' }}
                    content={<CustomTooltip />}
                  />
                  <Bar dataKey="MLP" name="MLP" animationDuration={400} radius={[2, 2, 0, 0]}>
                    {dayPredictions.map((entry, i) => (
                      <Cell
                        key={`mlp-${i}`}
                        fill="var(--accent)"
                        fillOpacity={entry.hour === inputs.hour ? 1 : 0.28}
                      />
                    ))}
                  </Bar>
                  <Bar dataKey="LR" name="LR" animationDuration={400} radius={[2, 2, 0, 0]}>
                    {dayPredictions.map((entry, i) => (
                      <Cell
                        key={`lr-${i}`}
                        fill="var(--lr-color)"
                        fillOpacity={entry.hour === inputs.hour ? 1 : 0.28}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="flex items-center gap-5 mt-3 text-xs text-[var(--text-muted)]">
              <LegendDot color="var(--accent)" label="MLP" />
              <LegendDot color="var(--lr-color)" label="Lineare Regression" />
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}

function LegendDot({ color, label }) {
  return (
    <span className="inline-flex items-center gap-2">
      <span
        className="inline-block w-2 h-2 rounded-full"
        style={{ background: color }}
      />
      {label}
    </span>
  )
}

// Wiederverwendbarer Schieberegler-Block mit Label und Wert-Anzeige
function Slider({ label, value, min, max, step = 1, onChange, display }) {
  return (
    <div>
      <div className="flex items-baseline justify-between mb-2.5">
        <label className="eyebrow">{label}</label>
        <span className="font-mono text-sm text-[var(--text-primary)]">{display}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={onChange}
      />
    </div>
  )
}

function Field({ label, children }) {
  return (
    <div>
      <div className="eyebrow mb-2.5">{label}</div>
      {children}
    </div>
  )
}
