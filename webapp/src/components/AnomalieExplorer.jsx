/**
 * Durchsucht alle stündlichen Vorhersageergebnisse nach grossen Fehlern,
 * systematischen Mustern und auffälligen Ausreissern für die Maturaarbeit.
 */
import { useState, useEffect, useMemo, useCallback } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip as RechartTooltip, ResponsiveContainer, ReferenceDot
} from 'recharts'

const PAGE_SIZE = 20
const TOP_N_OPTIONS = [5, 10, 20, 50]
const WEEKDAY_SHORT = ['Mo', 'Di', 'Mi', 'Do', 'Fr', 'Sa', 'So']
const WEEKDAY_FULL  = ['Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag', 'Sonntag']

const SEARCH_TYPES = [
  {
    id: 'top_errors',
    label: 'Grösste Abweichungen',
    desc: 'Die X Stunden mit dem höchsten absoluten Fehler'
  },
  {
    id: 'underestimate',
    label: 'Starke Unterschätzung',
    desc: 'Modell sagt deutlich weniger als tatsächlich gefahren'
  },
  {
    id: 'overestimate',
    label: 'Starke Überschätzung',
    desc: 'Modell sagt deutlich mehr als tatsächlich gefahren'
  },
  {
    id: 'peak_fail',
    label: 'Peak-Versagen',
    desc: 'Grosse Fehler während Morgen- oder Abendspitze (7–9 / 17–19 Uhr)'
  },
  {
    id: 'outlier',
    label: 'Relative Ausreisser',
    desc: 'Stunden mit überdurchschnittlich hohem prozentualen MLP-Fehler'
  },
  {
    id: 'weekday',
    label: 'Wochentag-Muster',
    desc: 'Fehler nach Wochentag filtern und vergleichen'
  }
]

// ── Hilfsfunktionen ──────────────────────────────────────────────────────────

function errColor(absErr) {
  if (absErr < 100)  return 'var(--success)'
  if (absErr <= 300) return 'var(--warning)'
  return 'var(--danger)'
}

function fmtDate(dt) {
  return dt.toLocaleDateString('de-CH', {
    weekday: 'long', day: '2-digit', month: 'long', year: 'numeric'
  })
}

function enrich(row, station) {
  const dt = new Date(row.datetime)
  const hour = dt.getHours()
  const jsDay = dt.getDay()
  const weekday = jsDay === 0 ? 6 : jsDay - 1
  const error_mlp    = row.pred_mlp    - row.actual
  const error_linear = row.pred_linear - row.actual
  return {
    ...row,
    stationId:  station.id,
    stationName: station.name,
    stationDir:  station.direction,
    dt,
    dateStr: row.datetime.slice(0, 10),
    hour,
    weekday,
    is_peak: (hour >= 7 && hour <= 9) || (hour >= 17 && hour <= 19),
    error_mlp,
    error_linear,
    abs_error_mlp:    Math.abs(error_mlp),
    abs_error_linear: Math.abs(error_linear),
    pct_error_mlp: row.actual > 0 ? (error_mlp / row.actual) * 100 : 0
  }
}

function buildContextText(row, allData) {
  const over   = row.error_mlp > 0
  const absErr = Math.round(row.abs_error_mlp)
  const pct    = Math.abs(row.pct_error_mlp).toFixed(1)
  const timeLabel = row.is_peak
    ? (row.hour < 12 ? 'Morgenspitze' : 'Abendspitze')
    : 'Nebenzeit'

  const top50 = [...allData]
    .sort((a, b) => b.abs_error_mlp - a.abs_error_mlp)
    .slice(0, 50)
  const rank = top50.findIndex(
    r => r.stationId === row.stationId && r.datetime === row.datetime
  )

  const rankPart =
    rank === 0 ? ' Dies ist die grösste Abweichung im gesamten Datensatz.' :
    rank > 0 && rank < 10 ? ` Dies gehört zu den ${rank + 1} grössten Abweichungen im Datensatz.` :
    rank >= 10 && rank < 50 ? ` Rang ${rank + 1} der grössten Abweichungen im Datensatz.` : ''

  return (
    `Das MLP ${over ? 'überschätzte' : 'unterschätzte'} den Verkehr in der ` +
    `${timeLabel} am ${fmtDate(row.dt)} um ${String(row.hour).padStart(2, '0')}:00 Uhr ` +
    `um ${absErr} Fahrzeuge (${over ? '+' : '−'}${pct} %).${rankPart}`
  )
}

function exportCsv(rows) {
  const header = 'datetime,station,richtung,actual,pred_mlp,pred_linear,error_mlp,error_linear,pct_error_mlp'
  const lines = rows.map(r =>
    [
      r.datetime, r.stationName, r.stationDir,
      r.actual, r.pred_mlp, r.pred_linear,
      r.error_mlp.toFixed(1), r.error_linear.toFixed(1),
      r.pct_error_mlp.toFixed(1)
    ].join(',')
  )
  const blob = new Blob([[header, ...lines].join('\n')], { type: 'text/csv;charset=utf-8;' })
  const url  = URL.createObjectURL(blob)
  const a    = document.createElement('a')
  a.href = url
  a.download = 'anomalien_export.csv'
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

// ── Daten-Hook ───────────────────────────────────────────────────────────────

function useAllResults(stations) {
  const [state, setState] = useState({ loading: true, data: [], error: null })

  useEffect(() => {
    if (!stations?.length) return
    let cancelled = false
    setState({ loading: true, data: [], error: null })

    Promise.all(
      stations.map(s =>
        fetch(`/data/results/${s.id}_daily.json`)
          .then(r => (r.ok ? r.json() : []))
          .then(rows => rows.map(row => enrich(row, s)))
          .catch(() => [])
      )
    ).then(chunks => {
      if (cancelled) return
      const all = chunks.flat()
      setState({ loading: false, data: all, error: all.length === 0 ? 'empty' : null })
    })

    return () => { cancelled = true }
  }, [stations])

  return state
}

// ── Sub-Komponenten ───────────────────────────────────────────────────────────

function SummaryCard({ label, value, unit, accent }) {
  return (
    <div className="card p-4 relative overflow-hidden">
      {accent && (
        <span
          aria-hidden
          className="absolute inset-y-0 left-0 w-px"
          style={{ background: accent }}
        />
      )}
      <div className="eyebrow mb-2 text-[0.63rem]">{label}</div>
      <div className="font-mono text-[1.35rem] leading-none text-[var(--text-primary)]">{value}</div>
      {unit && <div className="text-[0.7rem] text-[var(--text-muted)] mt-1.5">{unit}</div>}
    </div>
  )
}

function DayChart({ stationId, dateStr, hour: highlightHour, actual: highlightActual, allData }) {
  const dayRows = useMemo(
    () =>
      allData
        .filter(r => r.stationId === stationId && r.dateStr === dateStr)
        .sort((a, b) => a.hour - b.hour)
        .map(r => ({ hour: r.hour, actual: r.actual, mlp: r.pred_mlp, lr: r.pred_linear })),
    [allData, stationId, dateStr]
  )

  if (!dayRows.length) {
    return (
      <div className="text-xs text-[var(--text-muted)] py-6 text-center">
        Tagesdaten nicht verfügbar.
      </div>
    )
  }

  return (
    <div style={{ width: '100%', height: 190 }}>
      <ResponsiveContainer>
        <LineChart data={dayRows} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
          <CartesianGrid stroke="var(--border)" strokeDasharray="2 3" />
          <XAxis dataKey="hour" stroke="var(--text-muted)" tick={{ fontSize: 10 }} />
          <YAxis stroke="var(--text-muted)" tick={{ fontSize: 10 }} />
          <RechartTooltip
            cursor={{ stroke: 'var(--border)' }}
            content={({ active, payload, label: lbl }) => {
              if (!active || !payload?.length) return null
              return (
                <div className="chart-tooltip">
                  <div className="font-mono mb-1">{String(lbl).padStart(2, '0')}:00 Uhr</div>
                  {payload.map(p => (
                    <div key={p.dataKey} className="row">
                      <span className="label" style={{ color: p.color }}>{p.name}</span>
                      <span>{Math.round(p.value)}</span>
                    </div>
                  ))}
                </div>
              )
            }}
          />
          <Line type="monotone" dataKey="actual" name="Tatsächlich" stroke="var(--text-primary)" strokeWidth={2} dot={false} isAnimationActive={false} />
          <Line type="monotone" dataKey="mlp"    name="MLP"         stroke="var(--accent)"        strokeWidth={2} dot={false} isAnimationActive={false} />
          <Line type="monotone" dataKey="lr"     name="LR"          stroke="var(--lr-color)"      strokeWidth={1.5} strokeDasharray="5 4" dot={false} isAnimationActive={false} />
          <ReferenceDot
            x={highlightHour}
            y={highlightActual}
            r={5}
            fill="var(--text-primary)"
            stroke="var(--bg-base)"
            strokeWidth={2}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

function ResultCard({ row, isExpanded, onToggle, onGoToDatum, allData }) {
  const color = errColor(row.abs_error_mlp)
  const over  = row.error_mlp > 0

  return (
    <div className="card overflow-hidden">
      <div
        className="p-4 cursor-pointer hover:bg-[var(--bg-elevated)] transition-colors select-none"
        onClick={onToggle}
      >
        {/* Zeile 1: Datum/Zeit + Station */}
        <div className="flex items-start justify-between gap-4 mb-2">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-mono text-[0.82rem] text-[var(--text-primary)] whitespace-nowrap">
              {row.dateStr} · {String(row.hour).padStart(2, '0')}:00 Uhr
            </span>
            {row.is_peak && (
              <span className="text-[10px] eyebrow text-[var(--warning)]">Spitze</span>
            )}
          </div>
          <div className="text-right shrink-0">
            <div className="text-xs text-[var(--text-muted)] whitespace-nowrap">{row.stationName}</div>
            <div className="text-[10px] text-[var(--text-faint)]">{row.stationDir}</div>
          </div>
        </div>

        {/* Zeile 2: Zählwerte */}
        <div className="flex flex-wrap items-center gap-x-4 gap-y-0.5 text-xs mb-2.5">
          <span>
            <span className="text-[var(--text-muted)]">Tatsächlich </span>
            <span className="font-mono text-[var(--text-primary)]">{row.actual}</span>
          </span>
          <span>
            <span className="text-[var(--text-muted)]">MLP </span>
            <span className="font-mono" style={{ color: 'var(--accent)' }}>{row.pred_mlp}</span>
          </span>
          <span>
            <span className="text-[var(--text-muted)]">LR </span>
            <span className="font-mono" style={{ color: 'var(--lr-color)' }}>{row.pred_linear}</span>
          </span>
        </div>

        {/* Zeile 3: Fehler-Badge + Aktionen */}
        <div className="flex items-center justify-between gap-3">
          <span
            className="inline-block px-2.5 py-0.5 rounded-full text-[11px] font-mono whitespace-nowrap"
            style={{ background: color + '1a', color, border: `1px solid ${color}44` }}
          >
            {over ? '+' : '−'}{Math.round(row.abs_error_mlp)} Fzg
            &nbsp;({over ? '+' : '−'}{Math.abs(row.pct_error_mlp).toFixed(1)} %)
          </span>

          <div className="flex items-center gap-3 shrink-0">
            <button
              onClick={e => { e.stopPropagation(); onGoToDatum(row.stationId, row.dateStr) }}
              className="btn-link text-[11px]"
              title="In Datums-Analyse öffnen"
            >
              Tageskurve →
            </button>
            <span
              className="text-[var(--text-faint)] text-xs inline-block transition-transform duration-200"
              style={{ transform: isExpanded ? 'rotate(180deg)' : 'rotate(0deg)' }}
            >
              ▾
            </span>
          </div>
        </div>
      </div>

      {/* Detail-Accordion */}
      {isExpanded && (
        <div className="border-t border-[var(--border)] p-4 bg-[var(--bg-sunken)] space-y-4">
          <DayChart
            stationId={row.stationId}
            dateStr={row.dateStr}
            hour={row.hour}
            actual={row.actual}
            allData={allData}
          />
          <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
            {buildContextText(row, allData)}
          </p>
        </div>
      )}
    </div>
  )
}

// ── Hauptkomponente ───────────────────────────────────────────────────────────

export default function AnomalieExplorer({ stations, onGoToDatum }) {
  const { loading, data: allData, error: loadError } = useAllResults(stations)

  // Filter-Zustand
  const [selStations, setSelStations] = useState(new Set())
  const [model,       setModel]       = useState('mlp')
  const [dateFrom,    setDateFrom]    = useState('')
  const [dateTo,      setDateTo]      = useState('')
  const [searchType,  setSearchType]  = useState('top_errors')
  const [topN,        setTopN]        = useState(10)
  const [threshold,   setThreshold]   = useState(100)
  const [pctThresh,   setPctThresh]   = useState(20)
  const [selWeekdays, setSelWeekdays] = useState(new Set([0, 1, 2, 3, 4, 5, 6]))

  // Ergebnis-Zustand
  const [results,     setResults]     = useState(null)
  const [sortBy,      setSortBy]      = useState('error')
  const [page,        setPage]        = useState(1)
  const [expandedKey, setExpandedKey] = useState(null)

  // Alle Stationen initial selektieren
  useEffect(() => {
    if (stations?.length) setSelStations(new Set(stations.map(s => s.id)))
  }, [stations])

  const toggleStation = useCallback(id => {
    setSelStations(prev => {
      const next = new Set(prev)
      next.has(id) ? next.delete(id) : next.add(id)
      return next
    })
  }, [])

  const toggleWeekday = useCallback(d => {
    setSelWeekdays(prev => {
      const next = new Set(prev)
      next.has(d) ? next.delete(d) : next.add(d)
      return next
    })
  }, [])

  const handleSearch = useCallback(() => {
    if (!allData.length) return

    let rows = [...allData]

    // Stations-Filter
    if (selStations.size > 0 && selStations.size < stations.length) {
      rows = rows.filter(r => selStations.has(r.stationId))
    }

    // Datum-Filter
    if (dateFrom) rows = rows.filter(r => r.dateStr >= dateFrom)
    if (dateTo)   rows = rows.filter(r => r.dateStr <= dateTo)

    const getAbsErr = r => model === 'linear' ? r.abs_error_linear : r.abs_error_mlp
    const getErr    = r => model === 'linear' ? r.error_linear     : r.error_mlp

    switch (searchType) {
      case 'top_errors':
        rows = rows.sort((a, b) => getAbsErr(b) - getAbsErr(a)).slice(0, topN)
        break
      case 'underestimate':
        rows = rows.filter(r => getErr(r) < -threshold)
        break
      case 'overestimate':
        rows = rows.filter(r => getErr(r) > threshold)
        break
      case 'peak_fail':
        rows = rows.filter(r => r.is_peak && getAbsErr(r) > threshold)
        break
      case 'outlier':
        rows = rows.filter(r => Math.abs(r.pct_error_mlp) > pctThresh)
        break
      case 'weekday':
        if (selWeekdays.size > 0 && selWeekdays.size < 7) {
          rows = rows.filter(r => selWeekdays.has(r.weekday))
        }
        break
    }

    setResults(rows)
    setPage(1)
    setExpandedKey(null)
  }, [allData, selStations, stations, model, dateFrom, dateTo, searchType, topN, threshold, pctThresh, selWeekdays])

  const sortedResults = useMemo(() => {
    if (!results) return []
    const copy = [...results]
    if (sortBy === 'error')   copy.sort((a, b) => b.abs_error_mlp - a.abs_error_mlp)
    if (sortBy === 'date')    copy.sort((a, b) => a.dt - b.dt)
    if (sortBy === 'station') copy.sort((a, b) => a.stationName.localeCompare(b.stationName))
    return copy
  }, [results, sortBy])

  const totalPages = Math.max(1, Math.ceil(sortedResults.length / PAGE_SIZE))
  const pageRows   = sortedResults.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE)

  const metrics = useMemo(() => {
    if (!results?.length) return null
    const errs  = results.map(r => r.abs_error_mlp)
    const avg   = errs.reduce((s, e) => s + e, 0) / errs.length
    const worst = results.reduce((acc, r) => !acc || r.abs_error_mlp > acc.abs_error_mlp ? r : acc, null)
    return {
      total:     results.length,
      avg:       avg.toFixed(0),
      max:       Math.max(...errs).toFixed(0),
      worstHour: worst ? `${String(worst.hour).padStart(2, '0')}:00` : '–'
    }
  }, [results])

  // ── Loading / Error ────────────────────────────────────────────────────────

  if (loading) {
    return (
      <div className="card p-16 text-center inline-flex items-center justify-center w-full gap-3 text-sm text-[var(--text-muted)]">
        <span className="inline-block w-1.5 h-1.5 rounded-full bg-[var(--accent)] animate-pulse" />
        Stationsdaten werden geladen…
      </div>
    )
  }

  if (loadError === 'empty') {
    return (
      <div className="card p-10 max-w-md mx-auto text-center">
        <div className="eyebrow mb-2">Hinweis</div>
        <h3 className="mb-2">Noch keine Ergebnisse vorhanden</h3>
        <p className="text-sm text-[var(--text-muted)]">
          Führe zuerst <code>export_weights.py</code> aus, um die Datensätze zu generieren.
        </p>
      </div>
    )
  }

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="space-y-8">
      <div className="stagger-in">
        <h1 className="mb-2">Anomalie-Explorer</h1>
        <p className="text-sm text-[var(--text-secondary)] max-w-prose">
          Durchsuche alle stündlichen Vorhersageergebnisse nach grossen Fehlern,
          systematischen Mustern und auffälligen Ausreissern.
        </p>
      </div>

      <div className="grid lg:grid-cols-[35fr_65fr] gap-6 items-start">

        {/* ── Filterbereich ─────────────────────────────────────────────── */}
        <aside className="card p-5 space-y-6 stagger-in">
          <h2 className="text-base">Filter</h2>

          {/* Station */}
          <section>
            <div className="flex items-center justify-between mb-2">
              <div className="eyebrow">Station</div>
              <button
                className="text-[10px] text-[var(--accent)] hover:text-[var(--accent-strong)] transition-colors"
                onClick={() => setSelStations(new Set(stations.map(s => s.id)))}
              >
                Alle
              </button>
            </div>
            <div className="space-y-1.5 max-h-40 overflow-y-auto pr-1">
              {stations.map(s => (
                <label
                  key={s.id}
                  className="flex items-center gap-2.5 text-xs cursor-pointer text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
                >
                  <input
                    type="checkbox"
                    checked={selStations.has(s.id)}
                    onChange={() => toggleStation(s.id)}
                    className="accent-[var(--accent)] shrink-0"
                  />
                  <span className="truncate">{s.name} · {s.direction}</span>
                </label>
              ))}
            </div>
          </section>

          {/* Modell */}
          <section>
            <div className="eyebrow mb-2.5">Modell</div>
            <div className="radio-pill">
              {[['mlp', 'MLP'], ['linear', 'LR'], ['both', 'Beide']].map(([v, lbl]) => (
                <span key={v}>
                  <input type="radio" name="ae-model" id={`ae-m-${v}`} checked={model === v} onChange={() => setModel(v)} />
                  <label htmlFor={`ae-m-${v}`}>{lbl}</label>
                </span>
              ))}
            </div>
          </section>

          {/* Zeitraum */}
          <section>
            <div className="eyebrow mb-2.5">Zeitraum</div>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-[10px] text-[var(--text-muted)] block mb-1">Von</label>
                <input type="date" value={dateFrom} onChange={e => setDateFrom(e.target.value)} className="w-full text-xs" />
              </div>
              <div>
                <label className="text-[10px] text-[var(--text-muted)] block mb-1">Bis</label>
                <input type="date" value={dateTo}   onChange={e => setDateTo(e.target.value)}   className="w-full text-xs" />
              </div>
            </div>
          </section>

          {/* Suchtyp */}
          <section>
            <div className="eyebrow mb-3">Suchtyp</div>
            <div className="space-y-2.5">
              {SEARCH_TYPES.map(st => (
                <div key={st.id}>
                  <label className="flex items-start gap-2.5 cursor-pointer group">
                    <input
                      type="radio"
                      name="ae-searchType"
                      checked={searchType === st.id}
                      onChange={() => setSearchType(st.id)}
                      className="mt-0.5 shrink-0 accent-[var(--accent)]"
                    />
                    <div>
                      <div className={`text-xs font-medium transition-colors ${
                        searchType === st.id
                          ? 'text-[var(--text-primary)]'
                          : 'text-[var(--text-secondary)] group-hover:text-[var(--text-primary)]'
                      }`}>
                        {st.label}
                      </div>
                      <div className="text-[10px] text-[var(--text-muted)] leading-relaxed">{st.desc}</div>
                    </div>
                  </label>

                  {searchType === st.id && (
                    <div className="mt-2.5 ml-5 pl-3 border-l border-[var(--border)]">
                      {st.id === 'top_errors' && (
                        <div>
                          <div className="text-[10px] text-[var(--text-muted)] mb-1.5">Top N Ergebnisse</div>
                          <div className="flex gap-1.5 flex-wrap">
                            {TOP_N_OPTIONS.map(n => (
                              <button
                                key={n}
                                onClick={() => setTopN(n)}
                                className={`px-3 py-1 text-[11px] rounded-sm border transition-colors ${
                                  topN === n
                                    ? 'border-[var(--accent)] text-[var(--accent)] bg-[var(--accent-soft)]'
                                    : 'border-[var(--border)] text-[var(--text-muted)] hover:border-[var(--border-strong)] hover:text-[var(--text-primary)]'
                                }`}
                              >
                                {n}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}

                      {['underestimate', 'overestimate', 'peak_fail'].includes(st.id) && (
                        <div>
                          <div className="flex items-baseline justify-between mb-1.5">
                            <div className="text-[10px] text-[var(--text-muted)]">Schwellenwert</div>
                            <span className="font-mono text-[11px] text-[var(--text-primary)]">{threshold} Fzg</span>
                          </div>
                          <input
                            type="range" min={20} max={500} step={10}
                            value={threshold}
                            onChange={e => setThreshold(Number(e.target.value))}
                          />
                        </div>
                      )}

                      {st.id === 'outlier' && (
                        <div>
                          <div className="flex items-baseline justify-between mb-1.5">
                            <div className="text-[10px] text-[var(--text-muted)]">Min. relativer Fehler</div>
                            <span className="font-mono text-[11px] text-[var(--text-primary)]">{pctThresh} %</span>
                          </div>
                          <input
                            type="range" min={5} max={80} step={5}
                            value={pctThresh}
                            onChange={e => setPctThresh(Number(e.target.value))}
                          />
                        </div>
                      )}

                      {st.id === 'weekday' && (
                        <div>
                          <div className="text-[10px] text-[var(--text-muted)] mb-1.5">Wochentage</div>
                          <div className="flex gap-1 flex-wrap">
                            {WEEKDAY_SHORT.map((lbl, i) => (
                              <button
                                key={i}
                                onClick={() => toggleWeekday(i)}
                                className={`w-7 h-7 text-[10px] rounded-sm border transition-colors ${
                                  selWeekdays.has(i)
                                    ? 'border-[var(--accent)] text-[var(--accent)] bg-[var(--accent-soft)]'
                                    : 'border-[var(--border)] text-[var(--text-muted)] hover:border-[var(--border-strong)]'
                                }`}
                              >
                                {lbl}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </section>

          {/* Suchen-Button */}
          <button
            onClick={handleSearch}
            disabled={!allData.length}
            className="w-full py-2.5 rounded-sm text-sm font-medium transition-opacity"
            style={{
              background: 'var(--accent)',
              color: '#0c0c0d',
              opacity: allData.length ? 1 : 0.45,
              cursor: allData.length ? 'pointer' : 'not-allowed'
            }}
          >
            Suchen
          </button>
        </aside>

        {/* ── Ergebnisbereich ───────────────────────────────────────────── */}
        <section className="space-y-5 stagger-in" style={{ animationDelay: '60ms' }}>

          {/* Zusammenfassung */}
          {metrics && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <SummaryCard label="Treffer gesamt"     value={metrics.total}     accent="var(--accent)" />
              <SummaryCard label="Ø Abweichung"       value={metrics.avg}       unit="Fahrzeuge" />
              <SummaryCard label="Grösster Fehler"    value={metrics.max}       unit="Fahrzeuge" />
              <SummaryCard label="Schlechteste Stunde" value={metrics.worstHour} />
            </div>
          )}

          {/* Ergebnisliste */}
          {results !== null && (
            <>
              <div className="flex items-center justify-between gap-4 flex-wrap">
                <span className="text-xs text-[var(--text-muted)]">
                  {sortedResults.length === 0
                    ? 'Keine Treffer'
                    : `${sortedResults.length} Treffer${totalPages > 1 ? ` · Seite ${page} / ${totalPages}` : ''}`}
                </span>
                {sortedResults.length > 0 && (
                  <div className="flex items-center gap-2.5">
                    <label className="eyebrow text-[0.63rem]">Sortieren nach</label>
                    <select
                      value={sortBy}
                      onChange={e => { setSortBy(e.target.value); setPage(1) }}
                      className="text-xs py-1.5"
                    >
                      <option value="error">Grösster Fehler</option>
                      <option value="date">Datum</option>
                      <option value="station">Station</option>
                    </select>
                  </div>
                )}
              </div>

              {sortedResults.length === 0 ? (
                <div className="card p-10 text-center">
                  <div className="eyebrow mb-2">Keine Treffer</div>
                  <p className="text-sm text-[var(--text-muted)]">
                    Keine Einträge gefunden. Passe die Filterkriterien an.
                  </p>
                </div>
              ) : (
                <>
                  <div className="space-y-2">
                    {pageRows.map(row => {
                      const key = `${row.stationId}-${row.datetime}`
                      return (
                        <ResultCard
                          key={key}
                          row={row}
                          isExpanded={expandedKey === key}
                          onToggle={() => setExpandedKey(prev => prev === key ? null : key)}
                          onGoToDatum={onGoToDatum}
                          allData={allData}
                        />
                      )
                    })}
                  </div>

                  {totalPages > 1 && (
                    <div className="flex items-center justify-center gap-3 pt-1">
                      <button
                        onClick={() => setPage(p => Math.max(1, p - 1))}
                        disabled={page === 1}
                        className="btn-ghost text-xs px-4 py-1.5 disabled:opacity-40 disabled:cursor-not-allowed"
                      >
                        ← Zurück
                      </button>
                      <span className="font-mono text-xs text-[var(--text-muted)]">
                        {page} / {totalPages}
                      </span>
                      <button
                        onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                        disabled={page === totalPages}
                        className="btn-ghost text-xs px-4 py-1.5 disabled:opacity-40 disabled:cursor-not-allowed"
                      >
                        Weiter →
                      </button>
                    </div>
                  )}

                  <div className="flex justify-end pt-1">
                    <button onClick={() => exportCsv(sortedResults)} className="btn-ghost text-xs">
                      Als CSV exportieren ↓
                    </button>
                  </div>
                </>
              )}
            </>
          )}

          {/* Leerzustand vor erster Suche */}
          {results === null && (
            <div className="card p-14 text-center text-[var(--text-muted)]">
              <div className="text-[2rem] mb-3 opacity-20 font-mono">⌕</div>
              <p className="text-sm">Wähle Filter und klicke «Suchen», um Treffer anzuzeigen.</p>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}
