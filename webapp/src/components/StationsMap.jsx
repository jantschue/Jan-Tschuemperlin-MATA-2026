import { useMemo, useState } from 'react'
import { MapContainer, TileLayer, CircleMarker, Tooltip } from 'react-leaflet'

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
  if (r2 >= 0.7) return '#22c55e'
  if (r2 >= 0.6) return '#f59e0b'
  return '#ef4444'
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
      className={`px-3 py-2 text-xs uppercase tracking-wider text-[var(--text-muted)] cursor-pointer select-none hover:text-[var(--text-primary)] ${
        right ? 'text-right' : 'text-left'
      }`}
    >
      {children}
      {sortKey === k && (
        <span className="ml-1 text-[var(--accent)]">{sortDir === 'asc' ? '▲' : '▼'}</span>
      )}
    </th>
  )

  return (
    <div className="space-y-8">
      <section className="stagger-in">
        <h2 className="font-mono text-xl mb-1">Stationskarte</h2>
        <p className="text-sm text-[var(--text-muted)] mb-4">
          Kanton Schwyz · 10 Messstellen · Farbe nach MLP-R²
        </p>

        <div
          className="card overflow-hidden"
          style={{ height: '60vh', minHeight: 380 }}
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
              url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
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
                    color: active ? '#f0f0f5' : r2Hex(s.metrics.mlp.r2),
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
                    <div className="font-mono text-sm mb-1">{s.name}</div>
                    <div className="text-xs text-[var(--text-muted)] mb-2">
                      Richtung {s.direction}
                    </div>
                    <div className="grid grid-cols-3 gap-x-3 text-xs">
                      <div></div>
                      <div className="text-[var(--text-muted)]">LR</div>
                      <div className="text-[var(--accent)]">MLP</div>
                      <div className="text-[var(--text-muted)]">MAE</div>
                      <div>{s.metrics.linear.mae.toFixed(1)}</div>
                      <div className="text-[var(--text-primary)]">{s.metrics.mlp.mae.toFixed(1)}</div>
                      <div className="text-[var(--text-muted)]">RMSE</div>
                      <div>{s.metrics.linear.rmse.toFixed(1)}</div>
                      <div className="text-[var(--text-primary)]">{s.metrics.mlp.rmse.toFixed(1)}</div>
                      <div className="text-[var(--text-muted)]">R²</div>
                      <div>{s.metrics.linear.r2.toFixed(2)}</div>
                      <div className="text-[var(--text-primary)]">{s.metrics.mlp.r2.toFixed(2)}</div>
                    </div>
                  </Tooltip>
                </CircleMarker>
              )
            })}
          </MapContainer>
        </div>
      </section>

      <section className="stagger-in" style={{ animationDelay: '80ms' }}>
        <h3 className="font-mono text-sm uppercase tracking-wider text-[var(--text-muted)] mb-3">
          Übersicht aller Stationen
        </h3>
        <div className="card overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="border-b border-[var(--border)]">
              <tr>
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
                    className={`border-b border-[var(--border)] last:border-b-0 cursor-pointer hover:bg-[var(--bg-elevated)] ${
                      active ? 'bg-[var(--bg-elevated)]' : ''
                    }`}
                  >
                    <td className="px-3 py-2">{s.name}</td>
                    <td className="px-3 py-2 text-[var(--text-muted)]">{s.direction}</td>
                    <td className="px-3 py-2 text-right font-mono">{s.metrics.linear.mae.toFixed(1)}</td>
                    <td className="px-3 py-2 text-right font-mono" style={{ color: r2Color(s.metrics.linear.r2) }}>
                      {s.metrics.linear.r2.toFixed(2)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono">{s.metrics.mlp.mae.toFixed(1)}</td>
                    <td className="px-3 py-2 text-right font-mono" style={{ color: r2Color(s.metrics.mlp.r2) }}>
                      {s.metrics.mlp.r2.toFixed(2)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono" style={{ color: imp >= 0 ? 'var(--success)' : 'var(--danger)' }}>
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
