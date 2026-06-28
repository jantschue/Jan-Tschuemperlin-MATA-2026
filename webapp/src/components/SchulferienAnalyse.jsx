/**
 * Schulferien-Analyse: zeigt, welche Schwyzer Schulferien im Testzeitraum liegen,
 * wie stark der Verkehr während dieser mehrwöchigen Perioden vom normalen
 * Wochentags-Durchschnitt abweicht, wie gut das Modell die Ferien trifft und
 * welchen isolierten Effekt das Schulferien-Feature hat. v8 ergänzt 26
 * kantonsspezifische Schulferienspalten (`schoolholiday_*`); das Counterfactual
 * ("keine Schulferien") wird in export_weights.py vorberechnet (alle
 * `schoolholiday_*`-Spalten auf 0) und hier nur noch dargestellt.
 *
 * Aufbau bewusst parallel zur Feiertags-Analyse, jedoch auf Ferien-Perioden
 * (Sport-, Frühlings-, Sommer-, Herbst-, Weihnachtsferien) statt Einzeltage.
 */
import { useMemo, useState, useEffect } from 'react'
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Cell
} from 'recharts'
import {
  useDailyResults, useSchoolSchedule, useSchoolHolidayFeatures, useAllDailyResults
} from '../hooks/useModelWeights.js'
import { holidayDateSet } from '../utils/swissHolidays.js'
import { schoolPeriods, schoolDateMap, schoolDateSet, SCHOOL_ORDER } from '../utils/swissSchoolHolidays.js'

const WEEKDAY_FULL = ['Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag', 'Sonntag']

// ── Hilfsfunktionen ───────────────────────────────────────────────────────────

function pad2(n) { return String(n).padStart(2, '0') }
function toWeekday(jsDay) { return jsDay === 0 ? 6 : jsDay - 1 }
function sumBy(arr, key) { return arr.reduce((s, r) => s + r[key], 0) }

// Datum als de-CH-String (z.B. "1. November 2024")
function fmtDate(dateStr) {
  const [y, m, d] = dateStr.split('-').map(Number)
  return new Date(y, m - 1, d).toLocaleDateString('de-CH', {
    day: 'numeric', month: 'long', year: 'numeric'
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

// Wochentag×Stunde-Referenzprofil über alle Tage ausserhalb von excludeSet
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

// Aggregiert je Ferien-Vorkommen die im Testset abgedeckten Tage zu Kennzahlen
function periodOccurrences(byDate, reference, periods) {
  const out = []
  for (const p of periods) {
    let nDays = 0, actualDailySum = 0, mlpDailySum = 0, refDailySum = 0
    for (const ds of p.dates) {
      const g = byDate.get(ds)
      if (!g) continue
      nDays += 1
      actualDailySum += sumBy(g.hours, 'actual')
      mlpDailySum += sumBy(g.hours, 'mlp')
      refDailySum += reference.daily[g.weekday]
    }
    if (nDays === 0) continue
    const actualDaily = actualDailySum / nDays
    const mlpDaily = mlpDailySum / nDays
    const refDaily = refDailySum / nDays
    out.push({
      id: p.id, name: p.name, year: p.year, start: p.start, end: p.end,
      coverageDays: nDays, actualDaily, mlpDaily, refDaily,
      diffRealPct: refDaily ? (actualDaily - refDaily) / refDaily * 100 : 0,
      diffModelPct: actualDaily ? (mlpDaily - actualDaily) / actualDaily * 100 : 0
    })
  }
  return out
}

// Stations-Statistik für das Ranking (über alle Ferien-Vorkommen der Station)
function stationStats(rows, periods, dateMap, schoolSet, years) {
  const byDate = groupByDate(rows)
  const yrs = years || [...new Set([...byDate.keys()].map(d => Number(d.slice(0, 4))))]
  const excludeSet = new Set([...holidayDateSet(yrs), ...schoolSet])
  const reference = buildReference(byDate, excludeSet)

  const occ = periodOccurrences(byDate, reference, periods)  // { name, diffRealPct }

  const schoolErr = []     // { name, absErr }
  let nonSchoolAbsSum = 0, nonSchoolN = 0
  for (const g of byDate.values()) {
    const meta = dateMap.get(g.dateStr)
    if (meta) {
      for (const h of g.hours) schoolErr.push({ name: meta.name, absErr: Math.abs(h.actual - h.mlp) })
    } else if (!excludeSet.has(g.dateStr)) {
      for (const h of g.hours) { nonSchoolAbsSum += Math.abs(h.actual - h.mlp); nonSchoolN += 1 }
    }
  }
  return { occ, schoolErr, nonSchoolAbsSum, nonSchoolN }
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

export default function SchulferienAnalyse({ station, stations }) {
  const daily = useDailyResults(station.id)
  const schedule = useSchoolSchedule()
  const schoolCf = useSchoolHolidayFeatures(station.id)

  const [selectedYear, setSelectedYear] = useState('all')
  const [selectedId, setSelectedId] = useState(null)
  const [sort, setSort] = useState({ col: 'start', dir: 'asc' })
  const [scope, setScope] = useState('current')

  const allDaily = useAllDailyResults(stations, scope === 'all')

  // Alle Ferien-Vorkommen aus dem gemeinsamen Kalender
  const allPeriods = useMemo(() => schoolPeriods(schedule.data), [schedule.data])
  const dateMap = useMemo(() => schoolDateMap(schedule.data), [schedule.data])
  const schoolSet = useMemo(() => schoolDateSet(schedule.data), [schedule.data])

  // Jahre aus dem Testset
  const years = useMemo(() => {
    if (!daily.data) return []
    return [...new Set(daily.data.map(r => Number(r.datetime.slice(0, 4))))].sort()
  }, [daily.data])

  useEffect(() => {
    if (years.length && selectedYear !== 'all' && !years.includes(selectedYear)) {
      setSelectedYear(years[years.length - 1])
    }
  }, [years]) // eslint-disable-line react-hooks/exhaustive-deps

  const byDate = useMemo(() => (daily.data ? groupByDate(daily.data) : new Map()), [daily.data])

  // Referenz schliesst Feiertage UND Schulferien aus (nur normale Schulwochen)
  const refExcludeSet = useMemo(
    () => new Set([...holidayDateSet(years), ...schoolSet]),
    [years, schoolSet]
  )
  const reference = useMemo(() => buildReference(byDate, refExcludeSet), [byDate, refExcludeSet])

  // Counterfactual-Werte nach Datum → Stunde
  const cfByDate = useMemo(() => {
    const m = new Map()
    for (const r of (schoolCf.data || [])) {
      const ds = r.datetime.slice(0, 10)
      const hour = Number(r.datetime.slice(11, 13))
      if (!m.has(ds)) m.set(ds, new Map())
      m.get(ds).set(hour, r.mlpNo)
    }
    return m
  }, [schoolCf.data])

  // Ferien-Vorkommen mit Kennzahlen (Jahr-Filter)
  const occurrences = useMemo(() => {
    const all = periodOccurrences(byDate, reference, allPeriods)
    return selectedYear === 'all' ? all : all.filter(o => o.year === selectedYear)
  }, [byDate, reference, allPeriods, selectedYear])

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

  useEffect(() => {
    if (!occurrences.length) { setSelectedId(null); return }
    if (!occurrences.some(o => o.id === selectedId)) {
      setSelectedId(occurrences[0].id)
    }
  }, [occurrences]) // eslint-disable-line react-hooks/exhaustive-deps

  const selected = useMemo(
    () => occurrences.find(o => o.id === selectedId) || null,
    [occurrences, selectedId]
  )
  const selectedPeriod = useMemo(
    () => allPeriods.find(p => p.id === selectedId) || null,
    [allPeriods, selectedId]
  )

  // 24h-Durchschnittsprofil der gewählten Ferien-Periode (über alle abgedeckten Tage)
  const cfData = useMemo(() => {
    if (!selectedPeriod) return []
    const acc = Array.from({ length: 24 }, () => ({
      aS: 0, aN: 0, mS: 0, mN: 0, nS: 0, nN: 0, rS: 0, rN: 0
    }))
    for (const ds of selectedPeriod.dates) {
      const g = byDate.get(ds)
      if (!g) continue
      const cfHours = cfByDate.get(ds)
      for (const h of g.hours) {
        const a = acc[h.hour]
        a.aS += h.actual; a.aN += 1
        a.mS += h.mlp; a.mN += 1
        a.rS += reference.prof[g.weekday][h.hour]; a.rN += 1
        if (cfHours && cfHours.has(h.hour)) { a.nS += cfHours.get(h.hour); a.nN += 1 }
      }
    }
    return acc.map((a, hour) => ({
      hour,
      actual: a.aN ? a.aS / a.aN : null,
      predSchool: a.mN ? a.mS / a.mN : null,
      predNoSchool: a.nN ? a.nS / a.nN : null,
      reference: a.rN ? Math.round(a.rS / a.rN) : null
    }))
  }, [selectedPeriod, byDate, cfByDate, reference])

  const cfMetrics = useMemo(() => {
    if (!cfData.length) return null
    let maeS = 0, maeN = 0, flagS = 0, flagN = 0, specS = 0, specN = 0
    for (const d of cfData) {
      if (d.actual != null && d.predSchool != null) { maeS += Math.abs(d.actual - d.predSchool); maeN++ }
      if (d.predSchool != null && d.predNoSchool != null) { flagS += Math.abs(d.predSchool - d.predNoSchool); flagN++ }
      if (d.actual != null && d.reference != null) { specS += Math.abs(d.actual - d.reference); specN++ }
    }
    return {
      mae: maeN ? maeS / maeN : 0,
      flag: flagN ? flagS / flagN : 0,
      special: specN ? specS / specN : 0,
      hasCf: flagN > 0
    }
  }, [cfData])

  // ── Ranking ───────────────────────────────────────────────────────────────
  const ranking = useMemo(() => {
    if (!allPeriods.length) return null
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

    const realByName = new Map()
    const maeByName = new Map()
    let nsSum = 0, nsN = 0

    for (const rows of stationRowSets) {
      const st = stationStats(rows, allPeriods, dateMap, schoolSet)
      for (const o of st.occ) {
        const e = realByName.get(o.name) || { sum: 0, n: 0 }
        e.sum += o.diffRealPct; e.n += 1; realByName.set(o.name, e)
      }
      for (const h of st.schoolErr) {
        const e = maeByName.get(h.name) || { sum: 0, n: 0 }
        e.sum += h.absErr; e.n += 1; maeByName.set(h.name, e)
      }
      nsSum += st.nonSchoolAbsSum; nsN += st.nonSchoolN
    }

    const order = (name) => {
      const i = SCHOOL_ORDER.indexOf(name)
      return i === -1 ? 99 : i
    }
    const realRank = [...realByName.entries()]
      .map(([name, e]) => ({ name, value: e.sum / e.n }))
      .sort((a, b) => order(a.name) - order(b.name))
    const maeRank = [...maeByName.entries()]
      .map(([name, e]) => ({ name, mae: e.sum / e.n }))
      .sort((a, b) => order(a.name) - order(b.name))

    return { realRank, maeRank, nonSchoolMAE: nsN ? nsSum / nsN : 0 }
  }, [scope, allDaily.data, daily.data, allPeriods, dateMap, schoolSet])

  // ── Render-Guards ──────────────────────────────────────────────────────────
  if (daily.loading || schedule.loading) {
    return (
      <div className="card p-16 text-center text-sm text-[var(--text-muted)] inline-flex items-center justify-center w-full gap-3">
        <span className="inline-block w-1.5 h-1.5 rounded-full bg-[var(--accent)] animate-pulse" />
        Schulferiendaten werden geladen
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

      {/* Kopf + Filter */}
      <div className="flex flex-wrap items-end justify-between gap-6 stagger-in">
        <div>
          <h1 className="mb-2">
            <span className="text-[var(--text-secondary)]">Wie</span> schlägt sich das Modell in den Schulferien?
          </h1>
          <p className="text-sm text-[var(--text-secondary)]">
            {station.name} · {station.direction}
            <span className="text-[var(--text-muted)]"> — Station oben rechts wechselbar</span>
          </p>
        </div>

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
      </div>

      {/* Ferien-Tabelle */}
      <section className="space-y-3 stagger-in" style={{ animationDelay: '40ms' }}>
        <div className="flex items-baseline justify-between gap-4 flex-wrap">
          <h2 className="text-base">Schulferien im Zeitraum</h2>
          <span className="text-xs text-[var(--text-muted)]">
            {occurrences.length} Perioden · Zeile wählen für Details
          </span>
        </div>

        {occurrences.length === 0 ? (
          <div className="card p-10 text-center text-sm text-[var(--text-muted)]">
            Keine Schulferien im gewählten Zeitraum mit vorhandenen Daten.
          </div>
        ) : (
          <>
            {/* Desktop-Tabelle */}
            <div className="card overflow-hidden hidden md:block">
              <table className="w-full text-sm tabular">
                <thead>
                  <tr className="text-left text-[var(--text-muted)] border-b border-[var(--border)]">
                    {[
                      ['name', 'Ferien'], ['start', 'Zeitraum'], ['coverageDays', 'Tage'],
                      ['actualDaily', 'Ist Ø/Tag'], ['mlpDaily', 'Modell Ø/Tag'], ['refDaily', 'Wochentag-Ø'],
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
                    const active = o.id === selectedId
                    return (
                      <tr
                        key={o.id}
                        onClick={() => setSelectedId(o.id)}
                        className={`border-b border-[var(--border)] last:border-0 cursor-pointer transition-colors ${
                          active ? 'bg-[var(--accent-soft)]' : 'hover:bg-[var(--bg-elevated)]'
                        }`}
                      >
                        <td className="px-4 py-3 text-[var(--text-primary)] whitespace-nowrap">{o.name}</td>
                        <td className="px-4 py-3 font-mono text-[var(--text-secondary)] whitespace-nowrap text-xs">
                          {o.start} – {o.end}
                        </td>
                        <td className="px-4 py-3 font-mono text-[var(--text-muted)]">{o.coverageDays}</td>
                        <td className="px-4 py-3 font-mono">{Math.round(o.actualDaily)}</td>
                        <td className="px-4 py-3 font-mono" style={{ color: 'var(--accent)' }}>{Math.round(o.mlpDaily)}</td>
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
                const active = o.id === selectedId
                return (
                  <button
                    key={o.id}
                    onClick={() => setSelectedId(o.id)}
                    className={`card w-full text-left p-4 transition-colors ${active ? 'bg-[var(--accent-soft)]' : ''}`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-[var(--text-primary)]">{o.name}</span>
                      <span className="font-mono text-xs text-[var(--text-muted)]">{o.start} – {o.end}</span>
                    </div>
                    <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs">
                      <span><span className="text-[var(--text-muted)]">Ist/Tag </span><span className="font-mono">{Math.round(o.actualDaily)}</span></span>
                      <span><span className="text-[var(--text-muted)]">Modell/Tag </span><span className="font-mono" style={{ color: 'var(--accent)' }}>{Math.round(o.mlpDaily)}</span></span>
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

      {/* Tagesprofil-Vergleich / Counterfactual */}
      {selected && (
        <section className="space-y-4 stagger-in">
          <div>
            <div className="eyebrow mb-1">Durchschnittliches Tagesprofil</div>
            <h2 className="text-base">{selected.name} {selected.year} · {fmtDate(selected.start)} – {fmtDate(selected.end)}</h2>
          </div>

          <div className="card p-6">
            {schoolCf.loading ? (
              <div className="py-16 text-center text-sm text-[var(--text-muted)]">Counterfactual wird geladen…</div>
            ) : (
              <div style={{ width: '100%', height: 360 }}>
                <ResponsiveContainer>
                  <LineChart data={cfData} margin={{ top: 16, right: 16, bottom: 0, left: -16 }}>
                    <CartesianGrid stroke="var(--border)" strokeDasharray="2 3" />
                    <XAxis dataKey="hour" stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
                    <YAxis stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
                    <Tooltip content={<CfTooltip />} cursor={{ stroke: 'var(--border)' }} />
                    <Line type="monotone" dataKey="reference" name="Wochentag-Referenz (Schulwoche)" stroke="var(--text-faint)" strokeWidth={1.5} strokeDasharray="2 3" dot={false} isAnimationActive={false} connectNulls />
                    <Line type="monotone" dataKey="actual" name="Tatsächlich (Ø)" stroke="var(--text-primary)" strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />
                    <Line type="monotone" dataKey="predSchool" name="Modell (Schulferien)" stroke="var(--accent)" strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />
                    <Line type="monotone" dataKey="predNoSchool" name="Counterfactual (keine Schulferien)" stroke="var(--lr-color)" strokeWidth={1.5} strokeDasharray="5 4" dot={false} isAnimationActive={false} connectNulls />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
            {!schoolCf.loading && cfMetrics && !cfMetrics.hasCf && (
              <p className="text-xs text-[var(--text-muted)] mt-3">
                Für diese Periode liegen keine Counterfactual-Werte vor — der isolierte Effekt des
                Schulferien-Flags kann nicht berechnet werden (Ist-Werte und Referenz bleiben sichtbar).
              </p>
            )}
          </div>

          {cfMetrics && (
            <div className="grid sm:grid-cols-3 gap-4">
              <MetricCard label="MLP MAE (Ø Tagesprofil)" value={cfMetrics.mae.toFixed(1)} unit="Fahrzeuge/h" accent="var(--accent)" />
              <MetricCard label="Effekt des Flags" value={cfMetrics.flag.toFixed(1)} unit="Ø |mit − ohne| Fzg/h" accent="var(--lr-color)" />
              <MetricCard label="Speziell vs. Schulwoche" value={cfMetrics.special.toFixed(1)} unit="Ø |Ist − Referenz| Fzg/h" />
            </div>
          )}
        </section>
      )}

      {/* Aggregierte Sicht / Ranking */}
      <section className="space-y-4 stagger-in">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <div className="eyebrow mb-1">Ranking</div>
            <h2 className="text-base">Welche Schulferien verändern den Verkehr am stärksten?</h2>
          </div>
          <div className="radio-pill">
            {[['current', 'Aktuelle Station'], ['all', 'Alle Stationen']].map(([v, lbl]) => (
              <span key={v}>
                <input type="radio" name="sf-scope" id={`sf-scope-${v}`} checked={scope === v} onChange={() => setScope(v)} />
                <label htmlFor={`sf-scope-${v}`}>{lbl}</label>
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
                Ø Abweichung vom gleichen Wochentag in normalen Schulwochen. Grün = mehr, Rot = weniger Verkehr.
              </p>
              <div style={{ width: '100%', height: Math.max(220, ranking.realRank.length * 34) }}>
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
              <h3 className="text-[0.95rem] mb-1.5">Modellgüte je Ferien</h3>
              <p className="text-xs text-[var(--text-muted)] mb-4">
                MAE in diesen Ferien. Linie = Ø MAE in normalen Schulwochen ({Math.round(ranking.nonSchoolMAE)}).
              </p>
              <div style={{ width: '100%', height: Math.max(220, ranking.maeRank.length * 34) }}>
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
                      x={ranking.nonSchoolMAE}
                      stroke="var(--warning)"
                      strokeDasharray="3 3"
                      label={{ value: 'Ø Schulwochen', position: 'top', fill: 'var(--warning)', fontSize: 10 }}
                    />
                    <Bar dataKey="mae" animationDuration={400} radius={[0, 2, 2, 0]}>
                      {ranking.maeRank.map((d, i) => (
                        <Cell key={i} fill={d.mae > ranking.nonSchoolMAE ? 'var(--danger)' : 'var(--accent)'} />
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
