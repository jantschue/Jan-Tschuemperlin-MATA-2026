/**
 * Feiertags-Analyse: zeigt, welche Schwyzer Feiertage im Testzeitraum vorkommen,
 * wie das Modell an diesen Tagen abschneidet, wie stark es auf die Feiertags-
 * Features reagiert (Counterfactual) und welche Feiertage den Verkehr am stärksten
 * verändern. Das v8-Modell hat 26 kantonsspezifische Feiertagsspalten statt eines
 * is_holiday-Flags; das Counterfactual setzt für den gewählten Tag alle 26
 * holiday_*-Spalten auf 0 ("ganz normaler Tag"). Die zusätzlichen v8-Schulferien-
 * Spalten (schoolholiday_*) bleiben dabei unverändert.
 */
import { useMemo, useState, useEffect } from 'react'
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Cell
} from 'recharts'
import {
  useDailyResults, useModelWeights, useHolidayFeatures, useAllDailyResults
} from '../hooks/useModelWeights.js'
import { mlpForward } from '../utils/modelForward.js'
import { holidayMap, holidayDateSet, HOLIDAY_ORDER } from '../utils/swissHolidays.js'

const WEEKDAY_FULL = ['Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag', 'Sonntag']

// ── Hilfsfunktionen ───────────────────────────────────────────────────────────

function pad2(n) { return String(n).padStart(2, '0') }
function toWeekday(jsDay) { return jsDay === 0 ? 6 : jsDay - 1 }
function sumBy(arr, key) { return arr.reduce((s, r) => s + r[key], 0) }

// Datum als de-CH-String (z.B. "Mittwoch, 1. November 2024")
function fmtDate(dateStr) {
  const [y, m, d] = dateStr.split('-').map(Number)
  return new Date(y, m - 1, d).toLocaleDateString('de-CH', {
    weekday: 'long', day: 'numeric', month: 'long', year: 'numeric'
  })
}

// Stündliche Zeilen nach Datum gruppieren
function groupByDate(rows) {
  const map = new Map()
  for (const r of rows) {
    const ds = r.datetime.slice(0, 10)
    let g = map.get(ds)
    if (!g) {
      const d = new Date(r.datetime)
      g = { dateStr: ds, weekday: toWeekday(d.getDay()), hours: [] }
      map.set(ds, g)
    }
    g.hours.push({
      hour: new Date(r.datetime).getHours(),
      actual: r.actual, mlp: r.pred_mlp, lr: r.pred_linear
    })
  }
  for (const g of map.values()) g.hours.sort((a, b) => a.hour - b.hour)
  return map
}

// Wochentag×Stunde-Referenzprofil über alle Nicht-Feiertage
function buildReference(byDate, excludeSet) {
  const acc = Array.from({ length: 7 }, () => Array.from({ length: 24 }, () => ({ s: 0, n: 0 })))
  for (const g of byDate.values()) {
    if (excludeSet.has(g.dateStr)) continue
    for (const h of g.hours) {
      const c = acc[g.weekday][h.hour]
      c.s += h.actual; c.n += 1
    }
  }
  const prof = acc.map(row => row.map(c => (c.n ? c.s / c.n : 0)))
  const daily = prof.map(row => row.reduce((s, v) => s + v, 0))
  return { prof, daily }
}

// Stations-Statistik für das Ranking (über alle Jahre der Station)
function stationStats(rows, includeBerchtoldstag) {
  const byDate = groupByDate(rows)
  const years = [...new Set([...byDate.keys()].map(d => Number(d.slice(0, 4))))]
  const excludeSet = holidayDateSet(years)
  const reference = buildReference(byDate, excludeSet)
  const dispMap = holidayMap(years, { includeOptional: includeBerchtoldstag })

  const occ = []           // { name, diffRealPct }
  const holidayErr = []    // { name, absErr }
  let nonHolAbsSum = 0, nonHolN = 0

  for (const g of byDate.values()) {
    const meta = dispMap.get(g.dateStr)
    if (meta) {
      const actualSum = sumBy(g.hours, 'actual')
      const refDaily = reference.daily[g.weekday]
      occ.push({ name: meta.name, diffRealPct: refDaily ? (actualSum - refDaily) / refDaily * 100 : 0 })
      for (const h of g.hours) holidayErr.push({ name: meta.name, absErr: Math.abs(h.actual - h.mlp) })
    } else if (!excludeSet.has(g.dateStr)) {
      for (const h of g.hours) { nonHolAbsSum += Math.abs(h.actual - h.mlp); nonHolN += 1 }
    }
  }
  return { occ, holidayErr, nonHolAbsSum, nonHolN }
}

// Farb-Codierung: Vorzeichen (Verkehrsänderung) bzw. Betrag (Modellgüte)
function signColor(pct) {
  if (Math.abs(pct) < 5) return 'var(--text-muted)'
  return pct > 0 ? 'var(--success)' : 'var(--danger)'
}
function magColor(pct) {
  const a = Math.abs(pct)
  if (a < 10) return 'var(--text-muted)'
  if (a < 25) return 'var(--warning)'
  return 'var(--danger)'
}
function fmtPct(pct) {
  return `${pct >= 0 ? '+' : '−'}${Math.abs(pct).toFixed(1)} %`
}

// ── Sub-Komponenten ─────────────────────────────────────────────────────────

function MetricCard({ label, value, unit, accent }) {
  return (
    <div className="card p-5 relative overflow-hidden">
      {accent && <span aria-hidden className="absolute inset-y-0 left-0 w-px" style={{ background: accent }} />}
      <div className="eyebrow mb-3">{label}</div>
      <div className="metric-display text-[1.7rem] leading-none">{value}</div>
      {unit && <div className="text-xs text-[var(--text-muted)] mt-2">{unit}</div>}
    </div>
  )
}

function CfTooltip({ active, payload, label }) {
  if (!active || !payload || !payload.length) return null
  return (
    <div className="chart-tooltip">
      <div className="font-mono mb-1">{pad2(label)}:00 Uhr</div>
      {payload.map(p => (
        <div key={p.dataKey} className="row">
          <span className="label" style={{ color: p.color }}>{p.name}</span>
          <span>{p.value == null ? '–' : Math.round(p.value)}</span>
        </div>
      ))}
    </div>
  )
}

// ── Hauptkomponente ───────────────────────────────────────────────────────────

export default function FeiertagsAnalyse({ station, stations }) {
  const daily = useDailyResults(station.id)
  const weights = useModelWeights(station.id)
  const holidayFeatures = useHolidayFeatures(station.id)

  const [selectedYear, setSelectedYear] = useState('all')
  const [includeBerchtoldstag, setIncludeBerchtoldstag] = useState(false)
  const [selectedDate, setSelectedDate] = useState(null)
  const [sort, setSort] = useState({ col: 'dateStr', dir: 'asc' })
  const [scope, setScope] = useState('current')

  const allDaily = useAllDailyResults(stations, scope === 'all')

  // Jahre aus dem Testset
  const years = useMemo(() => {
    if (!daily.data) return []
    return [...new Set(daily.data.map(r => Number(r.datetime.slice(0, 10).slice(0, 4))))].sort()
  }, [daily.data])

  // Standardjahr setzen, sobald die Daten geladen sind
  useEffect(() => {
    if (years.length && selectedYear !== 'all' && !years.includes(selectedYear)) {
      setSelectedYear(years[years.length - 1])
    }
  }, [years]) // eslint-disable-line react-hooks/exhaustive-deps

  const byDate = useMemo(() => (daily.data ? groupByDate(daily.data) : new Map()), [daily.data])

  // Referenz schliesst immer alle Feiertage (inkl. Berchtoldstag) aus
  const refExcludeSet = useMemo(() => holidayDateSet(years), [years])
  const reference = useMemo(() => buildReference(byDate, refExcludeSet), [byDate, refExcludeSet])

  // Anzuzeigende Feiertage (Jahr-Filter + Berchtoldstag-Toggle)
  const displayMap = useMemo(() => {
    const yrs = selectedYear === 'all' ? years : [selectedYear]
    return holidayMap(yrs, { includeOptional: includeBerchtoldstag })
  }, [years, selectedYear, includeBerchtoldstag])

  // Feiertags-Vorkommen mit Kennzahlen
  const occurrences = useMemo(() => {
    const out = []
    for (const [ds, meta] of displayMap) {
      const g = byDate.get(ds)
      if (!g) continue
      const actualSum = sumBy(g.hours, 'actual')
      const mlpSum = sumBy(g.hours, 'mlp')
      const refDaily = reference.daily[g.weekday]
      out.push({
        name: meta.name, category: meta.category, optional: !!meta.optional,
        dateStr: ds, weekday: g.weekday,
        actualSum, mlpSum, refDaily,
        diffRealPct: refDaily ? (actualSum - refDaily) / refDaily * 100 : 0,
        diffModelPct: actualSum ? (mlpSum - actualSum) / actualSum * 100 : 0,
        coverage: g.hours.length
      })
    }
    return out
  }, [displayMap, byDate, reference])

  // Sortierte Tabellenzeilen
  const sortedRows = useMemo(() => {
    const copy = [...occurrences]
    const { col, dir } = sort
    copy.sort((a, b) => {
      let va = a[col], vb = b[col]
      if (typeof va === 'string') { va = va.toLowerCase(); vb = vb.toLowerCase() }
      if (va < vb) return dir === 'asc' ? -1 : 1
      if (va > vb) return dir === 'asc' ? 1 : -1
      return 0
    })
    return copy
  }, [occurrences, sort])

  // Auswahl synchron halten
  useEffect(() => {
    if (!occurrences.length) { setSelectedDate(null); return }
    if (!occurrences.some(o => o.dateStr === selectedDate)) {
      setSelectedDate(occurrences[0].dateStr)
    }
  }, [occurrences]) // eslint-disable-line react-hooks/exhaustive-deps

  const selected = useMemo(
    () => occurrences.find(o => o.dateStr === selectedDate) || null,
    [occurrences, selectedDate]
  )

  // Feiertags-Feature-Vektoren nach Datum
  const featByDate = useMemo(() => {
    const m = new Map()
    for (const r of (holidayFeatures.data || [])) {
      const ds = r.datetime.slice(0, 10)
      const hour = new Date(r.datetime).getHours()
      if (!m.has(ds)) m.set(ds, new Map())
      m.get(ds).set(hour, r.f)
    }
    return m
  }, [holidayFeatures.data])

  // Counterfactual-Verlauf für den gewählten Tag (24 Stunden)
  const cfData = useMemo(() => {
    if (!selected || !weights.weights) return []
    // Indizes aller Feiertagsspalten (holiday_*) im stationsspezifischen
    // Feature-Vektor – für das Counterfactual ("kein Feiertag") auf 0 gesetzt.
    const holidayIdx = (weights.weights.mlp.features || [])
      .map((n, i) => (n.startsWith('holiday_') ? i : -1))
      .filter((i) => i >= 0)
    const featHours = featByDate.get(selected.dateStr) || new Map()
    const dayHours = byDate.get(selected.dateStr)?.hours || []
    const actualByHour = new Map(dayHours.map(h => [h.hour, h.actual]))
    const out = []
    for (let h = 0; h < 24; h++) {
      const f = featHours.get(h)
      let predHoliday = null, predNoHoliday = null
      if (f) {
        predHoliday = mlpForward(f, weights.weights.mlp)
        const f0 = [...f]; for (const idx of holidayIdx) f0[idx] = 0
        predNoHoliday = mlpForward(f0, weights.weights.mlp)
      }
      out.push({
        hour: h,
        actual: actualByHour.has(h) ? actualByHour.get(h) : null,
        predHoliday, predNoHoliday,
        reference: Math.round(reference.prof[selected.weekday][h])
      })
    }
    return out
  }, [selected, weights.weights, featByDate, byDate, reference])

  const cfMetrics = useMemo(() => {
    if (!cfData.length) return null
    let maeS = 0, maeN = 0, flagS = 0, flagN = 0, specS = 0, specN = 0
    for (const d of cfData) {
      if (d.actual != null && d.predHoliday != null) { maeS += Math.abs(d.actual - d.predHoliday); maeN++ }
      if (d.predHoliday != null && d.predNoHoliday != null) { flagS += Math.abs(d.predHoliday - d.predNoHoliday); flagN++ }
      if (d.actual != null) { specS += Math.abs(d.actual - d.reference); specN++ }
    }
    return {
      mae: maeN ? maeS / maeN : 0,
      flag: flagN ? flagS / flagN : 0,
      special: specN ? specS / specN : 0,
      hasCf: flagN > 0
    }
  }, [cfData])

  // ── Ranking (Abschnitt 4) ────────────────────────────────────────────────
  const ranking = useMemo(() => {
    let stationRowSets
    if (scope === 'all') {
      if (!allDaily.data) return null
      const byStation = new Map()
      for (const r of allDaily.data) {
        if (!byStation.has(r.stationId)) byStation.set(r.stationId, [])
        byStation.get(r.stationId).push(r)
      }
      stationRowSets = [...byStation.values()]
    } else {
      if (!daily.data) return null
      stationRowSets = [daily.data]
    }

    const realByName = new Map() // name -> { sum, n }
    const maeByName = new Map()  // name -> { sum, n }
    let nhSum = 0, nhN = 0

    for (const rows of stationRowSets) {
      const st = stationStats(rows, includeBerchtoldstag)
      for (const o of st.occ) {
        const e = realByName.get(o.name) || { sum: 0, n: 0 }
        e.sum += o.diffRealPct; e.n += 1; realByName.set(o.name, e)
      }
      for (const h of st.holidayErr) {
        const e = maeByName.get(h.name) || { sum: 0, n: 0 }
        e.sum += h.absErr; e.n += 1; maeByName.set(h.name, e)
      }
      nhSum += st.nonHolAbsSum; nhN += st.nonHolN
    }

    const order = (name) => {
      const i = HOLIDAY_ORDER.indexOf(name)
      return i === -1 ? 99 : i
    }
    const realRank = [...realByName.entries()]
      .map(([name, e]) => ({ name, value: e.sum / e.n }))
      .sort((a, b) => order(a.name) - order(b.name))
    const maeRank = [...maeByName.entries()]
      .map(([name, e]) => ({ name, mae: e.sum / e.n }))
      .sort((a, b) => order(a.name) - order(b.name))

    return { realRank, maeRank, nonHolidayMAE: nhN ? nhSum / nhN : 0 }
  }, [scope, allDaily.data, daily.data, includeBerchtoldstag])

  // ── Render-Guards ──────────────────────────────────────────────────────────
  if (daily.loading) {
    return (
      <div className="card p-16 text-center text-sm text-[var(--text-muted)] inline-flex items-center justify-center w-full gap-3">
        <span className="inline-block w-1.5 h-1.5 rounded-full bg-[var(--accent)] animate-pulse" />
        Feiertagsdaten werden geladen
      </div>
    )
  }
  if (daily.error || !daily.data || daily.data.length === 0) {
    return (
      <div className="card p-10 max-w-md mx-auto text-center">
        <div className="eyebrow mb-2">Hinweis</div>
        <h3 className="mb-2">Daten fehlen</h3>
        <p className="text-sm text-[var(--text-muted)]">
          Tagesergebnisse für Station <span className="font-mono">{station.id}</span> noch nicht exportiert.
        </p>
      </div>
    )
  }

  const sortArrow = (col) => sort.col === col ? (sort.dir === 'asc' ? ' ▲' : ' ▼') : ''
  const setSortCol = (col) =>
    setSort(s => s.col === col ? { col, dir: s.dir === 'asc' ? 'desc' : 'asc' } : { col, dir: 'asc' })

  return (
    <div className="space-y-10">

      {/* Kopf + Filterleiste (Abschnitt 1) */}
      <div className="flex flex-wrap items-end justify-between gap-6 stagger-in">
        <div>
          <h1 className="mb-2">
            <span className="text-[var(--text-secondary)]">Wie</span> schlägt sich das Modell an Feiertagen?
          </h1>
          <p className="text-sm text-[var(--text-secondary)]">
            {station.name} · {station.direction}
            <span className="text-[var(--text-muted)]"> — Station oben rechts wechselbar</span>
          </p>
        </div>

        <div className="flex flex-wrap items-end gap-6">
          <div>
            <label className="eyebrow block mb-1.5">Jahr</label>
            <select
              value={selectedYear}
              onChange={(e) => setSelectedYear(e.target.value === 'all' ? 'all' : Number(e.target.value))}
            >
              <option value="all">Alle Jahre</option>
              {years.map(y => <option key={y} value={y}>{y}</option>)}
            </select>
          </div>
          <label className="flex items-center gap-3 text-sm text-[var(--text-secondary)] mb-1.5 cursor-pointer">
            <span>Berchtoldstag</span>
            <span className="toggle">
              <input
                type="checkbox"
                checked={includeBerchtoldstag}
                onChange={(e) => setIncludeBerchtoldstag(e.target.checked)}
              />
              <span className="slider" />
            </span>
          </label>
        </div>
      </div>

      {/* Feiertags-Tabelle (Abschnitt 2) */}
      <section className="space-y-3 stagger-in" style={{ animationDelay: '40ms' }}>
        <div className="flex items-baseline justify-between gap-4 flex-wrap">
          <h2 className="text-base">Feiertage im Zeitraum</h2>
          <span className="text-xs text-[var(--text-muted)]">
            {occurrences.length} Vorkommen · Zeile wählen für Details
          </span>
        </div>

        {occurrences.length === 0 ? (
          <div className="card p-10 text-center text-sm text-[var(--text-muted)]">
            Keine Feiertage im gewählten Zeitraum mit vorhandenen Daten.
          </div>
        ) : (
          <>
            {/* Desktop-Tabelle */}
            <div className="card overflow-hidden hidden md:block">
              <table className="w-full text-sm tabular">
                <thead>
                  <tr className="text-left text-[var(--text-muted)] border-b border-[var(--border)]">
                    {[
                      ['name', 'Feiertag'], ['dateStr', 'Datum'], ['weekday', 'Wochentag'],
                      ['actualSum', 'Ist (Tag)'], ['mlpSum', 'Modell (Tag)'], ['refDaily', 'Wochentag-Ø'],
                      ['diffRealPct', 'Δ Real vs. Ref'], ['diffModelPct', 'Δ Modell vs. Real']
                    ].map(([col, lbl]) => (
                      <th
                        key={col}
                        onClick={() => setSortCol(col)}
                        className="px-4 py-3 font-normal cursor-pointer select-none hover:text-[var(--text-primary)] transition-colors whitespace-nowrap eyebrow"
                      >
                        {lbl}{sortArrow(col)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sortedRows.map(o => {
                    const active = o.dateStr === selectedDate
                    return (
                      <tr
                        key={o.dateStr}
                        onClick={() => setSelectedDate(o.dateStr)}
                        className={`border-b border-[var(--border)] last:border-0 cursor-pointer transition-colors ${
                          active ? 'bg-[var(--accent-soft)]' : 'hover:bg-[var(--bg-elevated)]'
                        }`}
                      >
                        <td className="px-4 py-3 text-[var(--text-primary)] whitespace-nowrap">
                          {o.name}
                          {o.optional && <span className="ml-1.5 text-[10px] eyebrow text-[var(--text-faint)]">opt.</span>}
                        </td>
                        <td className="px-4 py-3 font-mono text-[var(--text-secondary)]">{o.dateStr}</td>
                        <td className="px-4 py-3 text-[var(--text-secondary)]">{WEEKDAY_FULL[o.weekday]}</td>
                        <td className="px-4 py-3 font-mono">{Math.round(o.actualSum)}</td>
                        <td className="px-4 py-3 font-mono" style={{ color: 'var(--accent)' }}>{Math.round(o.mlpSum)}</td>
                        <td className="px-4 py-3 font-mono text-[var(--text-muted)]">{Math.round(o.refDaily)}</td>
                        <td className="px-4 py-3 font-mono" style={{ color: signColor(o.diffRealPct) }}>{fmtPct(o.diffRealPct)}</td>
                        <td className="px-4 py-3 font-mono" style={{ color: magColor(o.diffModelPct) }}>{fmtPct(o.diffModelPct)}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>

            {/* Mobile-Karten */}
            <div className="md:hidden space-y-2">
              {sortedRows.map(o => {
                const active = o.dateStr === selectedDate
                return (
                  <button
                    key={o.dateStr}
                    onClick={() => setSelectedDate(o.dateStr)}
                    className={`card w-full text-left p-4 transition-colors ${active ? 'bg-[var(--accent-soft)]' : ''}`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-[var(--text-primary)]">{o.name}</span>
                      <span className="font-mono text-xs text-[var(--text-muted)]">{o.dateStr}</span>
                    </div>
                    <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs">
                      <span><span className="text-[var(--text-muted)]">Ist </span><span className="font-mono">{Math.round(o.actualSum)}</span></span>
                      <span><span className="text-[var(--text-muted)]">Modell </span><span className="font-mono" style={{ color: 'var(--accent)' }}>{Math.round(o.mlpSum)}</span></span>
                      <span><span className="text-[var(--text-muted)]">Δ Real </span><span className="font-mono" style={{ color: signColor(o.diffRealPct) }}>{fmtPct(o.diffRealPct)}</span></span>
                      <span><span className="text-[var(--text-muted)]">Δ Modell </span><span className="font-mono" style={{ color: magColor(o.diffModelPct) }}>{fmtPct(o.diffModelPct)}</span></span>
                    </div>
                  </button>
                )
              })}
            </div>
          </>
        )}
      </section>

      {/* Tagesprofil-Vergleich / Counterfactual (Abschnitt 3) */}
      {selected && (
        <section className="space-y-4 stagger-in">
          <div>
            <div className="eyebrow mb-1">Tagesprofil</div>
            <h2 className="text-base">{selected.name} · {fmtDate(selected.dateStr)}</h2>
          </div>

          <div className="card p-6">
            {weights.loading ? (
              <div className="py-16 text-center text-sm text-[var(--text-muted)]">Modell wird geladen…</div>
            ) : (
              <div style={{ width: '100%', height: 360 }}>
                <ResponsiveContainer>
                  <LineChart data={cfData} margin={{ top: 16, right: 16, bottom: 0, left: -16 }}>
                    <CartesianGrid stroke="var(--border)" strokeDasharray="2 3" />
                    <XAxis dataKey="hour" stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
                    <YAxis stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
                    <Tooltip content={<CfTooltip />} cursor={{ stroke: 'var(--border)' }} />
                    <Line type="monotone" dataKey="reference" name="Wochentag-Referenz" stroke="var(--text-faint)" strokeWidth={1.5} strokeDasharray="2 3" dot={false} isAnimationActive={false} connectNulls />
                    <Line type="monotone" dataKey="actual" name="Tatsächlich" stroke="var(--text-primary)" strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />
                    <Line type="monotone" dataKey="predHoliday" name="Modell (Feiertag)" stroke="var(--accent)" strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />
                    <Line type="monotone" dataKey="predNoHoliday" name="Counterfactual (kein Feiertag)" stroke="var(--lr-color)" strokeWidth={1.5} strokeDasharray="5 4" dot={false} isAnimationActive={false} connectNulls />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
            {!weights.loading && cfMetrics && !cfMetrics.hasCf && (
              <p className="text-xs text-[var(--text-muted)] mt-3">
                Für diesen Tag liegen keine Feiertags-Feature-Vektoren vor — das Counterfactual kann nicht
                berechnet werden (Ist-Werte und Wochentag-Referenz bleiben sichtbar).
              </p>
            )}
          </div>

          {cfMetrics && (
            <div className="grid sm:grid-cols-3 gap-4">
              <MetricCard label="MLP MAE (dieser Tag)" value={cfMetrics.mae.toFixed(1)} unit="Fahrzeuge/h" accent="var(--accent)" />
              <MetricCard label="Effekt des Flags" value={cfMetrics.flag.toFixed(1)} unit="Ø |mit − ohne| Fzg/h" accent="var(--lr-color)" />
              <MetricCard label="Speziell vs. Wochentag" value={cfMetrics.special.toFixed(1)} unit="Ø |Ist − Referenz| Fzg/h" />
            </div>
          )}
        </section>
      )}

      {/* Aggregierte Sicht / Ranking (Abschnitt 4) */}
      <section className="space-y-4 stagger-in">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <div className="eyebrow mb-1">Ranking</div>
            <h2 className="text-base">Welcher Feiertag verändert den Verkehr am stärksten?</h2>
          </div>
          <div className="radio-pill">
            {[['current', 'Aktuelle Station'], ['all', 'Alle Stationen']].map(([v, lbl]) => (
              <span key={v}>
                <input type="radio" name="fa-scope" id={`fa-scope-${v}`} checked={scope === v} onChange={() => setScope(v)} />
                <label htmlFor={`fa-scope-${v}`}>{lbl}</label>
              </span>
            ))}
          </div>
        </div>

        {scope === 'all' && allDaily.loading ? (
          <div className="card p-12 text-center text-sm text-[var(--text-muted)]">Alle Stationen werden geladen…</div>
        ) : ranking && ranking.realRank.length > 0 ? (
          <div className="grid md:grid-cols-2 gap-4">
            {/* Chart A: Verkehrsänderung real */}
            <div className="card p-5">
              <h3 className="text-[0.95rem] mb-1.5">Verkehrsänderung (real)</h3>
              <p className="text-xs text-[var(--text-muted)] mb-4">
                Ø Abweichung vom gleichen Wochentag (Nicht-Feiertage). Grün = mehr, Rot = weniger Verkehr.
              </p>
              <div style={{ width: '100%', height: Math.max(220, ranking.realRank.length * 30) }}>
                <ResponsiveContainer>
                  <BarChart data={ranking.realRank} layout="vertical" margin={{ left: 30, right: 16 }}>
                    <CartesianGrid stroke="var(--border)" strokeDasharray="2 3" horizontal={false} />
                    <XAxis type="number" stroke="var(--text-muted)" tick={{ fontSize: 11 }} tickFormatter={(v) => `${Math.round(v)}%`} />
                    <YAxis dataKey="name" type="category" stroke="var(--text-muted)" tick={{ fontSize: 11 }} width={120} />
                    <Tooltip
                      cursor={{ fill: 'rgba(255,255,255,0.03)' }}
                      content={({ active, payload }) => active && payload?.length ? (
                        <div className="chart-tooltip">
                          <div>{payload[0].payload.name}</div>
                          <div className="row"><span className="label">Δ Real</span><span>{fmtPct(payload[0].value)}</span></div>
                        </div>
                      ) : null}
                    />
                    <ReferenceLine x={0} stroke="var(--border-strong)" />
                    <Bar dataKey="value" animationDuration={400} radius={[0, 2, 2, 0]}>
                      {ranking.realRank.map((d, i) => <Cell key={i} fill={signColor(d.value)} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Chart B: Modellgüte (MAE) */}
            <div className="card p-5">
              <h3 className="text-[0.95rem] mb-1.5">Modellgüte je Feiertag</h3>
              <p className="text-xs text-[var(--text-muted)] mb-4">
                MAE an diesem Feiertag. Linie = Ø MAE an Nicht-Feiertagen ({Math.round(ranking.nonHolidayMAE)}).
              </p>
              <div style={{ width: '100%', height: Math.max(220, ranking.maeRank.length * 30) }}>
                <ResponsiveContainer>
                  <BarChart data={ranking.maeRank} layout="vertical" margin={{ left: 30, right: 16 }}>
                    <CartesianGrid stroke="var(--border)" strokeDasharray="2 3" horizontal={false} />
                    <XAxis type="number" stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
                    <YAxis dataKey="name" type="category" stroke="var(--text-muted)" tick={{ fontSize: 11 }} width={120} />
                    <Tooltip
                      cursor={{ fill: 'rgba(255,255,255,0.03)' }}
                      content={({ active, payload }) => active && payload?.length ? (
                        <div className="chart-tooltip">
                          <div>{payload[0].payload.name}</div>
                          <div className="row"><span className="label">MAE</span><span>{Math.round(payload[0].value)}</span></div>
                        </div>
                      ) : null}
                    />
                    <ReferenceLine
                      x={ranking.nonHolidayMAE}
                      stroke="var(--warning)"
                      strokeDasharray="3 3"
                      label={{ value: 'Ø Nicht-Feiertage', position: 'top', fill: 'var(--warning)', fontSize: 10 }}
                    />
                    <Bar dataKey="mae" animationDuration={400} radius={[0, 2, 2, 0]}>
                      {ranking.maeRank.map((d, i) => (
                        <Cell key={i} fill={d.mae > ranking.nonHolidayMAE ? 'var(--danger)' : 'var(--accent)'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        ) : (
          <div className="card p-10 text-center text-sm text-[var(--text-muted)]">Keine Ranking-Daten verfügbar.</div>
        )}
      </section>
    </div>
  )
}
