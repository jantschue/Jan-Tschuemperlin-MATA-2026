import { useMemo, useState, useEffect } from 'react'
import {
  Area,
  CartesianGrid,
  Line,
  ComposedChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts'
import { useDailyResults } from '../hooks/useModelWeights.js'

// Datum aus ISO-String herauslösen (YYYY-MM-DD)
function isoDate(dt) {
  return dt.slice(0, 10)
}

// MAE für eine Tages-Reihe berechnen
function mae(arr, key) {
  if (!arr.length) return 0
  const sum = arr.reduce((s, r) => s + Math.abs(r.actual - r[key]), 0)
  return sum / arr.length
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload || !payload.length) return null
  return (
    <div className="chart-tooltip">
      <div className="font-mono mb-1">{String(label).padStart(2, '0')}:00 Uhr</div>
      {payload
        .filter((p) => p.dataKey !== 'weekAvg')
        .map((p) => (
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

function MetricCard({ label, value, unit }) {
  return (
    <div className="card p-4">
      <div className="text-xs uppercase tracking-wider text-[var(--text-muted)] mb-2">
        {label}
      </div>
      <div className="font-mono text-2xl">{value}</div>
      {unit && <div className="text-xs text-[var(--text-muted)] mt-1">{unit}</div>}
    </div>
  )
}

export default function DatumsAnalyse({ station }) {
  const { loading, data, error } = useDailyResults(station.id)
  const [selectedDate, setSelectedDate] = useState('')
  const [showWeekAvg, setShowWeekAvg] = useState(false)

  // Beim ersten Laden ein Standarddatum aus dem Datensatz wählen
  useEffect(() => {
    if (data && data.length > 0 && !selectedDate) {
      const mid = data[Math.floor(data.length / 2)].datetime
      setSelectedDate(isoDate(mid))
    }
  }, [data, selectedDate])

  const dayRows = useMemo(() => {
    if (!data || !selectedDate) return []
    return data
      .filter((r) => isoDate(r.datetime) === selectedDate)
      .map((r) => ({
        hour: new Date(r.datetime).getHours(),
        actual: r.actual,
        mlp: r.pred_mlp,
        lr: r.pred_linear
      }))
      .sort((a, b) => a.hour - b.hour)
  }, [data, selectedDate])

  // Stündlicher Wochendurchschnitt über den gesamten Datensatz
  const weekAvg = useMemo(() => {
    if (!data) return new Map()
    const buckets = Array.from({ length: 24 }, () => ({ sum: 0, n: 0 }))
    for (const r of data) {
      const h = new Date(r.datetime).getHours()
      buckets[h].sum += r.actual
      buckets[h].n += 1
    }
    const map = new Map()
    buckets.forEach((b, h) => map.set(h, b.n ? b.sum / b.n : 0))
    return map
  }, [data])

  const chartData = useMemo(() => {
    return dayRows.map((r) => ({
      ...r,
      weekAvg: weekAvg.get(r.hour) || 0
    }))
  }, [dayRows, weekAvg])

  const mlpMae = mae(dayRows, 'mlp')
  const lrMae = mae(dayRows, 'lr')

  // Schlechteste und beste Stunde (MLP-Abweichung vom Ist-Wert)
  const errors = dayRows.map((r) => ({ ...r, err: r.mlp - r.actual }))
  const worst = errors.reduce(
    (acc, r) => (acc === null || Math.abs(r.err) > Math.abs(acc.err) ? r : acc),
    null
  )
  const best = errors.reduce(
    (acc, r) => (acc === null || Math.abs(r.err) < Math.abs(acc.err) ? r : acc),
    null
  )

  if (loading) {
    return (
      <div className="card p-12 text-center text-sm text-[var(--text-muted)]">
        Tagesdaten werden geladen...
      </div>
    )
  }

  if (error || !data || data.length === 0) {
    return (
      <div className="card p-8 max-w-md mx-auto text-center">
        <h3 className="font-mono mb-2">Daten fehlen</h3>
        <p className="text-sm text-[var(--text-muted)]">
          Tagesergebnisse für Station <span className="font-mono">{station.id}</span> noch
          nicht exportiert.
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="font-mono text-xl mb-1">Datums-Analyse</h2>
          <p className="text-sm text-[var(--text-muted)]">
            {station.name} · {station.direction}
          </p>
        </div>

        <div className="flex items-end gap-6">
          <div>
            <label className="block text-xs uppercase tracking-wider text-[var(--text-muted)] mb-1">
              Datum
            </label>
            <input
              type="date"
              value={selectedDate}
              onChange={(e) => setSelectedDate(e.target.value)}
            />
          </div>

          <label className="flex items-center gap-3 text-sm text-[var(--text-muted)] mb-2">
            <span>Wochendurchschnitt</span>
            <span className="toggle">
              <input
                type="checkbox"
                checked={showWeekAvg}
                onChange={(e) => setShowWeekAvg(e.target.checked)}
              />
              <span className="slider" />
            </span>
          </label>
        </div>
      </div>

      <div className="card p-6">
        {dayRows.length === 0 ? (
          <div className="py-16 text-center text-sm text-[var(--text-muted)]">
            Für dieses Datum sind keine Daten vorhanden.
          </div>
        ) : (
          <div style={{ width: '100%', height: 360 }}>
            <ResponsiveContainer>
              <ComposedChart data={chartData} margin={{ top: 16, right: 16, bottom: 0, left: -16 }}>
                <CartesianGrid stroke="var(--border)" strokeDasharray="2 3" />
                <XAxis dataKey="hour" stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
                <YAxis stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
                <Tooltip content={<CustomTooltip />} cursor={{ stroke: 'var(--border)' }} />

                {showWeekAvg && (
                  <Area
                    dataKey="weekAvg"
                    name="Wochendurchschnitt"
                    stroke="none"
                    fill="rgba(255,255,255,0.06)"
                    isAnimationActive={false}
                  />
                )}

                <Line
                  type="monotone"
                  dataKey="actual"
                  name="Tatsächlich"
                  stroke="var(--text-primary)"
                  strokeWidth={2}
                  dot={false}
                  animationDuration={400}
                />
                <Line
                  type="monotone"
                  dataKey="mlp"
                  name="MLP"
                  stroke="var(--accent)"
                  strokeWidth={2}
                  dot={false}
                  animationDuration={400}
                />
                <Line
                  type="monotone"
                  dataKey="lr"
                  name="Lineare Regression"
                  stroke="var(--lr-color)"
                  strokeWidth={1.5}
                  strokeDasharray="5 4"
                  dot={false}
                  animationDuration={400}
                />

                {worst && (
                  <ReferenceLine
                    x={worst.hour}
                    stroke="var(--warning)"
                    strokeDasharray="3 3"
                    label={{
                      value: 'Grösster MLP-Fehler',
                      position: 'top',
                      fill: 'var(--warning)',
                      fontSize: 11
                    }}
                  />
                )}
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard label="MLP MAE" value={mlpMae.toFixed(1)} unit="Fahrzeuge/h" />
        <MetricCard label="LR MAE" value={lrMae.toFixed(1)} unit="Fahrzeuge/h" />
        <MetricCard
          label="Schlechteste Stunde (MLP)"
          value={worst ? `${String(worst.hour).padStart(2, '0')}:00` : '–'}
          unit={worst ? `Abweichung ${worst.err >= 0 ? '+' : ''}${Math.round(worst.err)}` : ''}
        />
        <MetricCard
          label="Beste Stunde (MLP)"
          value={best ? `${String(best.hour).padStart(2, '0')}:00` : '–'}
          unit={best ? `Abweichung ${best.err >= 0 ? '+' : ''}${Math.round(best.err)}` : ''}
        />
      </div>

      {worst && (
        <div className="card p-4 text-sm text-[var(--text-muted)]">
          {(() => {
            const over = worst.err > 0
            const absErr = Math.abs(worst.err)
            const pct = worst.actual > 0 ? (absErr / worst.actual) * 100 : 0
            return (
              <>
                Um <span className="font-mono text-[var(--text-primary)]">
                  {String(worst.hour).padStart(2, '0')}:00 Uhr
                </span>{' '}
                {over ? 'überschätzt' : 'unterschätzt'} das Modell den Peak um{' '}
                <span className="font-mono text-[var(--text-primary)]">{absErr}</span>{' '}
                Fahrzeuge ({over ? '+' : '-'}
                {pct.toFixed(1)}%).
              </>
            )
          })()}
        </div>
      )}
    </div>
  )
}
