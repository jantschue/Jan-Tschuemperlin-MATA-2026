import { useMemo, useState } from 'react'
import { MapContainer, TileLayer, CircleMarker, Tooltip } from 'react-leaflet'
import { useTheme } from '../hooks/useTheme.js'

// Bounding-Box aller Stationen mit etwas Polsterung – sorgt dafür, dass beim
// Öffnen der Karte alle Marker komfortabel sichtbar sind.
function stationsBounds(stations) {
  if (stations.length === 0) return null
  let minLat = Infinity, maxLat = -Infinity, minLng = Infinity, maxLng = -Infinity
  for (const s of stations) {
    if (s.lat < minLat) minLat = s.lat
    if (s.lat > maxLat) maxLat = s.lat
    if (s.lng < minLng) minLng = s.lng
    if (s.lng > maxLng) maxLng = s.lng
  }
  const padLat = Math.max(0.02, (maxLat - minLat) * 0.15)
  const padLng = Math.max(0.02, (maxLng - minLng) * 0.15)
  return [
    [minLat - padLat, minLng - padLng],
    [maxLat + padLat, maxLng + padLng]
  ]
}

// Farbgebung nach R²-Schwellenwert
function r2Color(r2) {
  if (r2 >= 0.7) return 'var(--success)'
  if (r2 >= 0.6) return 'var(--warning)'
  return 'var(--danger)'
}

function r2Hex(r2) {
  if (r2 >= 0.7) return '#5fa776'
  if (r2 >= 0.6) return '#d9a24a'
  return '#d96a5c'
}

// Verbesserung in % MAE-Reduktion (positiv = MLP besser)
function improvement(metrics) {
  const lrMae = metrics.linear.mae
  const mlpMae = metrics.mlp.mae
  if (!lrMae) return 0
  return ((lrMae - mlpMae) / lrMae) * 100
}

export default function StationsMap({ stations, selectedStationId, onSelectStation }) {
  const [sortKey, setSortKey] = useState('name')
  const [sortDir, setSortDir] = useState('asc')
  const [theme] = useTheme()
  const isLight = theme === 'light'

  // Kachel-Layer passend zum Theme: helle CartoDB-Kacheln im Light-Mode,
  // dunkle im Dark-Mode. Der aktive Marker-Ring wird entsprechend kontrastiert.
  const tileUrl = isLight
    ? 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png'
    : 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
  const activeRing = isLight ? '#1b1b18' : '#f0f0f5'

  const bounds = useMemo(() => stationsBounds(stations), [stations])

  const sorted = useMemo(() => {
    const copy = [...stations]
    copy.sort((a, b) => {
      const getter = {
        name: (s) => s.name,
        direction: (s) => s.direction,
        lrMae: (s) => s.metrics.linear.mae,
        lrR2: (s) => s.metrics.linear.r2,
        mlpMae: (s) => s.metrics.mlp.mae,
        mlpR2: (s) => s.metrics.mlp.r2,
        improvement: (s) => improvement(s.metrics)
      }[sortKey]
      const av = getter(a)
      const bv = getter(b)
      if (av < bv) return sortDir === 'asc' ? -1 : 1
      if (av > bv) return sortDir === 'asc' ? 1 : -1
      return 0
    })
    return copy
  }, [stations, sortKey, sortDir])

  const onHeaderClick = (key) => {
    if (sortKey === key) setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    else {
      setSortKey(key)
      setSortDir('asc')
    }
  }

  const Header = ({ k, children, right }) => (
    <th
      onClick={() => onHeaderClick(k)}
      className={`px-4 py-3 eyebrow cursor-pointer select-none hover:text-[var(--text-primary)] transition-colors ${
        right ? 'text-right' : 'text-left'
      }`}
    >
      <span className="inline-flex items-center gap-1.5">
        {children}
        {sortKey === k && (
          <span className="text-[var(--accent)] text-[0.7rem]">
            {sortDir === 'asc' ? '↑' : '↓'}
          </span>
        )}
      </span>
    </th>
  )

  return (
    <div className="space-y-12">
      <section className="stagger-in">
        <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-4 mb-6">
          <div>
            <h1 className="mb-2">Stationskarte</h1>
            <p className="text-sm text-[var(--text-secondary)] max-w-prose">
              Zehn ASTRA-Messstellen im Kanton Schwyz. Markierungsfarbe codiert die
              MLP-Vorhersagegüte (R²) je Station.
            </p>
          </div>
          <div className="flex items-center gap-5 text-xs text-[var(--text-muted)]">
            <LegendDot color="var(--success)" label="R² ≥ 0.70" />
            <LegendDot color="var(--warning)" label="0.60–0.69" />
            <LegendDot color="var(--danger)" label="&lt; 0.60" />
          </div>
        </div>

        <div
          className="card overflow-hidden"
          style={{ height: 'min(60vh, 540px)', minHeight: 380, isolation: 'isolate' }}
        >
          <MapContainer
            bounds={bounds || undefined}
            center={bounds ? undefined : [47.09, 8.75]}
            zoom={bounds ? undefined : 11}
            scrollWheelZoom
            style={{ height: '100%', width: '100%', background: 'var(--bg-base)' }}
          >
            <TileLayer
              attribution='&copy; <a href="https://carto.com/">CartoDB</a>'
              url={tileUrl}
              subdomains="abcd"
            />
            {stations.map((s) => {
              const active = s.id === selectedStationId
              return (
                <CircleMarker
                  key={s.id}
                  center={[s.lat, s.lng]}
                  radius={active ? 11 : 7}
                  pathOptions={{
                    fillColor: r2Hex(s.metrics.mlp.r2),
                    fillOpacity: 0.95,
                    color: active ? activeRing : r2Hex(s.metrics.mlp.r2),
                    weight: active ? 2 : 1
                  }}
                  eventHandlers={{
                    click: () => onSelectStation(s.id)
                  }}
                >
                  <Tooltip
                    direction="top"
                    offset={[0, -8]}
                    opacity={1}
                    className="station-tooltip"
                  >
                    <div style={{ minWidth: 200 }}>
                      <div className="font-medium text-sm mb-0.5 whitespace-nowrap">{s.name}</div>
                      <div className="text-xs text-[var(--text-muted)] mb-3 whitespace-nowrap">
                        Richtung {s.direction}
                      </div>
                      <table className="text-xs w-full border-collapse">
                        <thead>
                          <tr>
                            <th className="font-normal text-left pb-1.5 text-[var(--text-faint)] pr-5" />
                            <th className="font-normal text-right pb-1.5 text-[var(--lr-color)] pr-4">LR</th>
                            <th className="font-normal text-right pb-1.5 text-[var(--accent)]">MLP</th>
                          </tr>
                        </thead>
                        <tbody>
                          {[
                            { label: 'MAE',  lr: s.metrics.linear.mae.toFixed(1),  mlp: s.metrics.mlp.mae.toFixed(1) },
                            { label: 'RMSE', lr: s.metrics.linear.rmse.toFixed(1), mlp: s.metrics.mlp.rmse.toFixed(1) },
                            { label: 'R²',   lr: s.metrics.linear.r2.toFixed(2),   mlp: s.metrics.mlp.r2.toFixed(2) }
                          ].map(row => (
                            <tr key={row.label}>
                              <td className="text-[var(--text-muted)] py-0.5 pr-5">{row.label}</td>
                              <td className="font-mono text-right whitespace-nowrap pr-4 text-[var(--text-secondary)]">{row.lr}</td>
                              <td className="font-mono text-right whitespace-nowrap text-[var(--text-primary)]">{row.mlp}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </Tooltip>
                </CircleMarker>
              )
            })}
          </MapContainer>
        </div>
      </section>

      <section className="stagger-in" style={{ animationDelay: '80ms' }}>
        <div className="flex items-baseline justify-between mb-4">
          <div>
            <h2>Übersicht aller Stationen</h2>
          </div>
          <span className="text-xs text-[var(--text-muted)] hidden sm:inline">
            Klick: zur Live-Vorhersage
          </span>
        </div>
        <div className="card-quiet overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[var(--border)]">
                <Header k="name">Station</Header>
                <Header k="direction">Richtung</Header>
                <Header k="lrMae" right>LR MAE</Header>
                <Header k="lrR2" right>LR R²</Header>
                <Header k="mlpMae" right>MLP MAE</Header>
                <Header k="mlpR2" right>MLP R²</Header>
                <Header k="improvement" right>Verbesserung</Header>
              </tr>
            </thead>
            <tbody>
              {sorted.map((s) => {
                const imp = improvement(s.metrics)
                const active = s.id === selectedStationId
                return (
                  <tr
                    key={s.id}
                    onClick={() => onSelectStation(s.id)}
                    className={`border-b border-[var(--border)] last:border-b-0 cursor-pointer transition-colors hover:bg-[var(--bg-elevated)] ${
                      active ? 'bg-[var(--bg-elevated)]' : ''
                    }`}
                  >
                    <td className="px-4 py-3 text-[var(--text-primary)]">
                      <span className="flex items-center gap-2.5">
                        <span
                          className="inline-block w-1.5 h-1.5 rounded-full"
                          style={{ background: r2Hex(s.metrics.mlp.r2) }}
                        />
                        {s.name}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-[var(--text-muted)]">{s.direction}</td>
                    <td className="px-4 py-3 text-right font-mono text-[var(--text-secondary)]">
                      {s.metrics.linear.mae.toFixed(1)}
                    </td>
                    <td
                      className="px-4 py-3 text-right font-mono"
                      style={{ color: r2Color(s.metrics.linear.r2) }}
                    >
                      {s.metrics.linear.r2.toFixed(2)}
                    </td>
                    <td className="px-4 py-3 text-right font-mono">
                      {s.metrics.mlp.mae.toFixed(1)}
                    </td>
                    <td
                      className="px-4 py-3 text-right font-mono"
                      style={{ color: r2Color(s.metrics.mlp.r2) }}
                    >
                      {s.metrics.mlp.r2.toFixed(2)}
                    </td>
                    <td
                      className="px-4 py-3 text-right font-mono"
                      style={{ color: imp >= 0 ? 'var(--success)' : 'var(--danger)' }}
                    >
                      {imp >= 0 ? '+' : ''}
                      {imp.toFixed(1)}%
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )
}

function LegendDot({ color, label }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span
        className="inline-block w-2 h-2 rounded-full"
        style={{ background: color }}
      />
      {label}
    </span>
  )
}
