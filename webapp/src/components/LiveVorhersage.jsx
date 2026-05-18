import { useMemo, useState } from 'react'
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

// Animations-Wrapper: rendert die Zahl bei Wert-Wechsel mit fade+slide-in
function AnimatedNumber({ value }) {
  return (
    <div key={value} className="animate-prediction font-mono text-[3.5rem] leading-none">
      {value}
    </div>
  )
}

function PredictionCard({ title, value, r2, color }) {
  return (
    <div className="card p-6 flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-mono text-sm tracking-wider uppercase text-[var(--text-muted)]">
          {title}
        </h3>
        <span
          className="text-xs font-mono px-2 py-1 rounded-sm"
          style={{ color, border: `1px solid ${color}` }}
        >
          R² = {r2.toFixed(2)}
        </span>
      </div>
      <AnimatedNumber value={value} />
      <div className="text-xs text-[var(--text-muted)] mt-2 tracking-wider uppercase">
        Fahrzeuge/h
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
  const [inputs, setInputs] = useState(DEFAULT_INPUTS)

  const update = (key) => (e) => {
    const val = e.target ? e.target.value : e
    setInputs((s) => ({ ...s, [key]: typeof val === 'string' ? Number(val) : val }))
  }

  // Vorhersage für aktuelle Eingaben
  const prediction = useMemo(() => {
    if (!weights) return { mlp: 0, linear: 0 }
    const v = buildFeatureVector(inputs)
    return {
      mlp: mlpForward(v, weights.mlp),
      linear: linearForward(v, weights.linear)
    }
  }, [inputs, weights])

  // 24h-Verlauf für das Balkendiagramm
  const dayPredictions = useMemo(() => {
    if (!weights) return []
    return Array.from({ length: 24 }, (_, h) => {
      const v = buildFeatureVector({ ...inputs, hour: h })
      return {
        hour: h,
        MLP: mlpForward(v, weights.mlp),
        LR: linearForward(v, weights.linear)
      }
    })
  }, [inputs, weights])

  if (loading) {
    return (
      <div className="card p-12 text-center">
        <div className="inline-block animate-pulse text-[var(--text-muted)] text-sm">
          Modell wird geladen...
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="card p-8 max-w-md mx-auto text-center">
        <h3 className="font-mono mb-2">Modell nicht verfügbar</h3>
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
    <div className="grid lg:grid-cols-5 gap-6">
      {/* Linkes Panel: Eingabeformular */}
      <aside className="lg:col-span-2 card p-6 stagger-in">
        <div className="flex items-center justify-between mb-1">
          <h2 className="font-mono text-lg">Eingaben</h2>
        </div>

        <div className="mb-6 pb-4 border-b border-[var(--border)]">
          <div className="text-xs uppercase tracking-wider text-[var(--text-muted)] mb-1">
            Station
          </div>
          <div className="flex items-center justify-between">
            <div className="font-mono text-sm">
              {station.name} · {station.direction}
            </div>
            <button
              onClick={onGoToMap}
              className="text-xs text-[var(--accent)] hover:underline"
            >
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
          />
          <PredictionCard
            title="Lineare Regression"
            value={prediction.linear}
            r2={lrR2}
            color="var(--lr-color)"
          />
        </div>

        <div className="card p-6">
          <div className="flex items-center justify-between mb-2">
            <h3 className="font-mono text-sm uppercase tracking-wider text-[var(--text-muted)]">
              24-Stunden Verlauf
            </h3>
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
                <YAxis stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
                <Tooltip
                  cursor={{ fill: 'rgba(255,255,255,0.03)' }}
                  content={<CustomTooltip />}
                />
                <Bar dataKey="MLP" name="MLP" animationDuration={400}>
                  {dayPredictions.map((entry, i) => (
                    <Cell
                      key={`mlp-${i}`}
                      fill={entry.hour === inputs.hour ? '#4f8ef7' : 'rgba(79,142,247,0.35)'}
                    />
                  ))}
                </Bar>
                <Bar dataKey="LR" name="LR" animationDuration={400}>
                  {dayPredictions.map((entry, i) => (
                    <Cell
                      key={`lr-${i}`}
                      fill={entry.hour === inputs.hour ? '#a78bfa' : 'rgba(167,139,250,0.3)'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>
    </div>
  )
}

// Wiederverwendbarer Schieberegler-Block mit Label und Wert-Anzeige
function Slider({ label, value, min, max, step = 1, onChange, display }) {
  return (
    <div>
      <div className="flex items-baseline justify-between mb-2">
        <label className="text-xs uppercase tracking-wider text-[var(--text-muted)]">
          {label}
        </label>
        <span className="font-mono text-sm">{display}</span>
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
      <div className="text-xs uppercase tracking-wider text-[var(--text-muted)] mb-2">
        {label}
      </div>
      {children}
    </div>
  )
}
