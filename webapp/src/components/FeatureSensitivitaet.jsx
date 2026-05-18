import { useMemo, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts'
import { useModelWeights } from '../hooks/useModelWeights.js'
import { mlpForward, linearForward } from '../utils/modelForward.js'
import {
  buildFeatureVector,
  DEFAULT_INPUTS,
  MONTHS,
  WEEKDAYS
} from '../utils/featureBuilder.js'

const SWEEPS = [
  { key: 'hour', label: 'Uhrzeit', unit: 'Uhr', min: 0, max: 23, step: 1, points: 24 },
  { key: 'temp', label: 'Temperatur', unit: '°C', min: -10, max: 35, points: 50 },
  { key: 'rain', label: 'Niederschlag', unit: 'mm', min: 0, max: 20, points: 50 },
  { key: 'sun', label: 'Sonnenstunden', unit: '', min: 0, max: 1, points: 50 }
]

// Lineare Werte-Reihe zwischen min und max
function linspace(min, max, n) {
  const out = new Array(n)
  const step = (max - min) / (n - 1)
  for (let i = 0; i < n; i++) out[i] = min + step * i
  return out
}

function CustomTooltip({ active, payload, label, sweep }) {
  if (!active || !payload || !payload.length) return null
  return (
    <div className="chart-tooltip">
      <div className="font-mono mb-1">
        {typeof label === 'number' ? label.toFixed(sweep.step === 1 ? 0 : 2) : label}
        {sweep.unit && ` ${sweep.unit}`}
      </div>
      {payload.map((p) => (
        <div key={p.dataKey} className="row">
          <span className="label" style={{ color: p.color }}>
            {p.name}
          </span>
          <span>{Math.round(p.value)}</span>
        </div>
      ))}
    </div>
  )
}

export default function FeatureSensitivitaet({ station }) {
  const { loading, weights, error } = useModelWeights(station.id)
  const [inputs, setInputs] = useState(DEFAULT_INPUTS)
  const [collapsed, setCollapsed] = useState(false)

  const update = (key) => (e) => {
    const val = e.target ? e.target.value : e
    setInputs((s) => ({ ...s, [key]: typeof val === 'string' ? Number(val) : val }))
  }

  // Sweeps für jedes der 4 Features berechnen
  const sweeps = useMemo(() => {
    if (!weights) return []
    return SWEEPS.map((sweep) => {
      const values = linspace(sweep.min, sweep.max, sweep.points)
      const data = values.map((v) => {
        const vec = buildFeatureVector({ ...inputs, [sweep.key]: v })
        return {
          x: v,
          MLP: mlpForward(vec, weights.mlp),
          LR: linearForward(vec, weights.linear)
        }
      })
      const mlpVals = data.map((d) => d.MLP)
      const sensitivity = Math.max(...mlpVals) - Math.min(...mlpVals)
      return { sweep, data, sensitivity }
    })
  }, [inputs, weights])

  const importance = useMemo(
    () =>
      sweeps
        .map((s) => ({ feature: s.sweep.label, sensitivity: s.sensitivity }))
        .sort((a, b) => b.sensitivity - a.sensitivity),
    [sweeps]
  )

  if (loading) {
    return (
      <div className="card p-12 text-center text-sm text-[var(--text-muted)]">
        Modell wird geladen...
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

  return (
    <div className="space-y-6">
      <div>
        <h2 className="font-mono text-xl mb-1">Feature-Sensitivität</h2>
        <p className="text-sm text-[var(--text-muted)]">
          {station.name} · {station.direction} – Wie reagiert das MLP auf
          Änderungen einzelner Features?
        </p>
      </div>

      {/* Basislinie (kollabierbar) */}
      <div className="card">
        <button
          className="w-full flex items-center justify-between px-6 py-4 text-left"
          onClick={() => setCollapsed((c) => !c)}
        >
          <div>
            <div className="font-mono text-sm uppercase tracking-wider text-[var(--text-muted)]">
              Basislinie
            </div>
            <div className="text-xs text-[var(--text-muted)] mt-1">
              Konstante Werte für alle nicht variierten Features
            </div>
          </div>
          <span className="text-[var(--text-muted)] text-sm">
            {collapsed ? '▾ Anzeigen' : '▴ Verbergen'}
          </span>
        </button>

        {!collapsed && (
          <div className="px-6 pb-6 grid sm:grid-cols-2 lg:grid-cols-4 gap-5 pt-2 border-t border-[var(--border)]">
            <CompactSlider
              label="Stunde"
              value={inputs.hour}
              min={0}
              max={23}
              onChange={update('hour')}
              display={`${String(inputs.hour).padStart(2, '0')}:00`}
            />
            <div>
              <Label>Wochentag</Label>
              <select value={inputs.dayOfWeek} onChange={update('dayOfWeek')} className="w-full">
                {WEEKDAYS.map((d, i) => (
                  <option key={i} value={i}>
                    {d}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <Label>Monat</Label>
              <select value={inputs.month} onChange={update('month')} className="w-full">
                {MONTHS.map((m, i) => (
                  <option key={i} value={i + 1}>
                    {m}
                  </option>
                ))}
              </select>
            </div>
            <CompactSlider
              label="Temperatur"
              value={inputs.temp}
              min={-10}
              max={35}
              onChange={update('temp')}
              display={`${inputs.temp}°C`}
            />
            <CompactSlider
              label="Niederschlag"
              value={inputs.rain}
              min={0}
              max={20}
              step={0.5}
              onChange={update('rain')}
              display={`${inputs.rain.toFixed(1)} mm`}
            />
            <CompactSlider
              label="Sonnenstunden"
              value={inputs.sun}
              min={0}
              max={1}
              step={0.05}
              onChange={update('sun')}
              display={inputs.sun.toFixed(2)}
            />
            <CompactSlider
              label="Schnee"
              value={inputs.snow}
              min={0}
              max={10}
              step={0.5}
              onChange={update('snow')}
              display={`${inputs.snow.toFixed(1)} mm`}
            />
            <div>
              <Label>Feiertag</Label>
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
            </div>
          </div>
        )}
      </div>

      {/* 2x2 Sensitivitäts-Grid */}
      <div className="grid md:grid-cols-2 gap-4">
        {sweeps.map(({ sweep, data }) => {
          const baseValue = inputs[sweep.key]
          return (
            <div key={sweep.key} className="card p-5">
              <div className="flex items-baseline justify-between mb-2">
                <h3 className="font-mono text-sm uppercase tracking-wider text-[var(--text-muted)]">
                  {sweep.label}
                </h3>
                <span className="text-xs text-[var(--text-muted)]">
                  Basis: {typeof baseValue === 'number' ? baseValue.toFixed(2) : baseValue}
                  {sweep.unit && ` ${sweep.unit}`}
                </span>
              </div>
              <div style={{ width: '100%', height: 220 }}>
                <ResponsiveContainer>
                  <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
                    <CartesianGrid stroke="var(--border)" strokeDasharray="2 3" />
                    <XAxis
                      dataKey="x"
                      type="number"
                      domain={[sweep.min, sweep.max]}
                      stroke="var(--text-muted)"
                      tick={{ fontSize: 11 }}
                      tickFormatter={(v) => (sweep.step === 1 ? v.toFixed(0) : v.toFixed(1))}
                    />
                    <YAxis stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
                    <Tooltip content={<CustomTooltip sweep={sweep} />} />
                    <Line
                      type="monotone"
                      dataKey="MLP"
                      stroke="var(--accent)"
                      strokeWidth={2}
                      dot={false}
                      animationDuration={400}
                    />
                    <Line
                      type="monotone"
                      dataKey="LR"
                      stroke="var(--lr-color)"
                      strokeWidth={1.5}
                      strokeDasharray="5 4"
                      dot={false}
                      animationDuration={400}
                    />
                    <ReferenceLine
                      x={baseValue}
                      stroke="var(--text-muted)"
                      strokeDasharray="2 2"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )
        })}
      </div>

      {/* Feature-Wichtigkeit */}
      <div className="card p-6">
        <h3 className="font-mono text-sm uppercase tracking-wider text-[var(--text-muted)] mb-1">
          Feature-Wichtigkeit
        </h3>
        <p className="text-xs text-[var(--text-muted)] mb-4">
          Wie stark verändert sich die Vorhersage wenn das Feature über seinen
          vollen Wertebereich variiert wird? (max - min)
        </p>
        <div style={{ width: '100%', height: 200 }}>
          <ResponsiveContainer>
            <BarChart data={importance} layout="vertical" margin={{ left: 12 }}>
              <CartesianGrid stroke="var(--border)" strokeDasharray="2 3" horizontal={false} />
              <XAxis type="number" stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
              <YAxis
                dataKey="feature"
                type="category"
                stroke="var(--text-muted)"
                tick={{ fontSize: 11 }}
                width={110}
              />
              <Tooltip
                cursor={{ fill: 'rgba(255,255,255,0.03)' }}
                content={({ active, payload }) =>
                  active && payload?.length ? (
                    <div className="chart-tooltip">
                      <div>{payload[0].payload.feature}</div>
                      <div className="row">
                        <span className="label">Sensitivität</span>
                        <span>{Math.round(payload[0].value)}</span>
                      </div>
                    </div>
                  ) : null
                }
              />
              <Bar dataKey="sensitivity" fill="var(--accent)" animationDuration={400} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}

function Label({ children }) {
  return (
    <div className="text-xs uppercase tracking-wider text-[var(--text-muted)] mb-2">
      {children}
    </div>
  )
}

function CompactSlider({ label, value, min, max, step = 1, onChange, display }) {
  return (
    <div>
      <div className="flex items-baseline justify-between mb-1">
        <Label>{label}</Label>
        <span className="font-mono text-xs">{display}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={onChange} />
    </div>
  )
}
