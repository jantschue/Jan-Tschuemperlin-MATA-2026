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

function MetricCard({ label, value, unit, accent }) {
  return (
    <div className="card p-5 relative overflow-hidden">
      {accent && (
        <span
          aria-hidden
          className="absolute inset-y-0 left-0 w-px"
          style={{ background: accent }}
        />
      )}
      <div className="eyebrow mb-3">{label}</div>
      <div className="metric-display text-[1.7rem] leading-none">{value}</div>
      {unit && (
        <div className="text-xs text-[var(--text-muted)] mt-2">{unit}</div>
      )}
    </div>
  )
}

export default function DatumsAnalyse({ station, initialDate = '' }) {
  const { loading, data, error } = useDailyResults(station.id)
  const [selectedDate, setSelectedDate] = useState(initialDate)

  // Synchronisiert wenn der Anomalie-Explorer ein neues Datum übergibt
  useEffect(() => {
    if (initialDate) setSelectedDate(initialDate)
  }, [initialDate])
  const [showWeekAvg, setShowWeekAvg] = useState(false)

  // Globales Y-Maximum über den gesamten Datensatz – bleibt bei Datumswechsel stabil.
  // Die Achse richtet sich an den echten Daten und dem MLP aus. Die lineare Baseline
  // kann durch Year-Extrapolation einzelne Ausreisser weit über den Daten erzeugen
  // (v.a. verkehrsstarke Stationen wie Wollerau/Wangen, LR bis ~7000 statt ~3000);
  // diese sollen die Skala nicht dominieren. Der LR erhält 15 % Spielraum über dem
  // Datenbereich, darüber hinausgehende Ausreisser werden oben abgeschnitten.
  const globalYMax = useMemo(() => {
    if (!data || !data.length) return 2000
    const dataMax = Math.max(...data.flatMap(r => [r.actual, r.pred_mlp]))
    const lrMax = Math.max(...data.map(r => r.pred_linear))
    const ymax = Math.max(dataMax, Math.min(lrMax, dataMax * 1.15))
    return Math.ceil(ymax / 100) * 100
  }, [data])

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
      <div className="card p-16 text-center text-sm text-[var(--text-muted)] inline-flex items-center justify-center w-full gap-3">
        <span className="inline-block w-1.5 h-1.5 rounded-full bg-[var(--accent)] animate-pulse" />
        Tagesdaten werden geladen
      </div>
    )
  }

  if (error || !data || data.length === 0) {
    return (
      <div className="card p-10 max-w-md mx-auto text-center">
        <div className="eyebrow mb-2">Hinweis</div>
        <h3 className="mb-2">Daten fehlen</h3>
        <p className="text-sm text-[var(--text-muted)]">
          Tagesergebnisse für Station <span className="font-mono">{station.id}</span> noch
          nicht exportiert.
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-end justify-between gap-6 stagger-in">
        <div>
          <h1 className="mb-2">
            <span className="text-[var(--text-secondary)]">Wo</span> liegt das Modell
            an einem konkreten Tag?
          </h1>
          <p className="text-sm text-[var(--text-secondary)]">
            {station.name} · {station.direction}
          </p>
        </div>

        <div className="flex flex-wrap items-end gap-6">
          <div>
            <label className="eyebrow block mb-1.5">Datum</label>
            <input
              type="date"
              value={selectedDate}
              onChange={(e) => setSelectedDate(e.target.value)}
            />
          </div>

          <label className="flex items-center gap-3 text-sm text-[var(--text-secondary)] mb-1.5 cursor-pointer">
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
                <YAxis domain={[0, globalYMax]} stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
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
        <MetricCard
          label="MLP MAE"
          value={mlpMae.toFixed(1)}
          unit="Fahrzeuge/h"
          accent="var(--accent)"
        />
        <MetricCard
          label="LR MAE"
          value={lrMae.toFixed(1)}
          unit="Fahrzeuge/h"
          accent="var(--lr-color)"
        />
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
        <div className="card-quiet p-5 text-sm text-[var(--text-secondary)] leading-relaxed">
          {(() => {
            const over = worst.err > 0
            const absErr = Math.abs(worst.err)
            const pct = worst.actual > 0 ? (absErr / worst.actual) * 100 : 0
            return (
              <>
                <span className="eyebrow mr-2">Auffällig</span>
                Um{' '}
                <span className="font-mono text-[var(--text-primary)]">
                  {String(worst.hour).padStart(2, '0')}:00 Uhr
                </span>{' '}
                {over ? 'überschätzt' : 'unterschätzt'} das Modell den Peak um{' '}
                <span className="font-mono text-[var(--text-primary)]">{absErr}</span>{' '}
                Fahrzeuge ({over ? '+' : '−'}
                {pct.toFixed(1)} %).
              </>
            )
          })()}
        </div>
      )}
    </div>
  )
}
