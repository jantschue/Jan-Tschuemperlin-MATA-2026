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
  const padLat = Math.max(0.02, (maxLat - minLat) * 0.12)
  const padLng = Math.max(0.02, (maxLng - minLng) * 0.25)
  return [
    [minLat - padLat, minLng - padLng],
    [maxLat + padLat, maxLng + padLng]
  ]
}

// Seite der Beschriftung je Standort. Brunnen und Schwyz liegen nur gut 2 km
// auseinander, ihre Beschriftungen zeigen deshalb in entgegengesetzte Richtungen.
const LABEL_SIDE = {
  '050_Brunnen_Mositunnel': 'left',
  '720_Schwyz': 'right',
  '171_Sattel': 'right',
  '299_Wollerau_Blatttunnel': 'left',
  '216_Wangen_SZ': 'right'
}

function groupByLocation(stations) {
  const map = new Map()
  for (const s of stations) {
    const key = s.id.replace(/_R\d+$/, '')
    if (!map.has(key)) map.set(key, { key, name: s.name, lat: 0, lng: 0, stations: [] })
    map.get(key).stations.push(s)
  }
  return [...map.values()].map((loc) => ({
    ...loc,
    lat: loc.stations.reduce((a, s) => a + s.lat, 0) / loc.stations.length,
    lng: loc.stations.reduce((a, s) => a + s.lng, 0) / loc.stations.length
  }))
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

  // Kachel-Layer: die graue Landeskarte von swisstopo. Sie ist Open Government
  // Data und braucht keinen API-Key (CartoDB verlangt inzwischen einen). Es gibt
  // nur diese eine Fassung, den Dark-Mode erzeugt index.css durch Invertieren
  // der Kachel-Ebene.
  const tileUrl =
    'https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-grau/default/current/3857/{z}/{x}/{y}.jpeg'
  const markerRing = isLight ? '#ffffff' : '#0b0b0d'
  const markerFill = isLight ? '#4861c9' : '#6c8ff5'

  // Beide Fahrtrichtungen einer Messstelle liegen praktisch am selben Ort.
  // Die Karte zeigt deshalb einen Punkt pro Standort, die Richtungen stehen
  // in der Beschriftung.
  const locations = useMemo(() => groupByLocation(stations), [stations])
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
        <div className="mb-6">
          <h1 className="mb-2">Stationskarte</h1>
          <p className="text-sm text-[var(--text-secondary)] max-w-prose">
            Fünf ASTRA-Messstellen im Kanton Schwyz, je zwei Fahrtrichtungen. Die
            Zahl neben der Richtung ist das R² des MLP auf den Testdaten. Ein Klick
            auf eine Richtung öffnet die Live-Vorhersage.
          </p>
        </div>

        <div
          className="card overflow-hidden"
          style={{ height: 'min(60vh, 540px)', minHeight: 380, isolation: 'isolate' }}
        >
          <MapContainer
            bounds={bounds || undefined}
            center={bounds ? undefined : [47.09, 8.75]}
            zoom={bounds ? undefined : 11}
            zoomSnap={0.25}
            scrollWheelZoom={false}
            minZoom={8}
            maxZoom={16}
            className="stations-map"
            style={{ height: '100%', width: '100%', background: 'var(--bg-base)' }}
          >
            <TileLayer
              attribution='&copy; <a href="https://www.swisstopo.admin.ch/">swisstopo</a>'
              url={tileUrl}
              minZoom={8}
              maxZoom={16}
            />
            {locations.map((loc) => {
              const active = loc.stations.some((s) => s.id === selectedStationId)
              return (
                <CircleMarker
                  key={loc.key}
                  center={[loc.lat, loc.lng]}
                  radius={active ? 9 : 7}
                  pathOptions={{
                    fillColor: markerFill,
                    fillOpacity: 1,
                    color: markerRing,
                    weight: 2.5
                  }}
                  eventHandlers={{
                    // Klicks in der Beschriftung leitet Leaflet ebenfalls an den
                    // Marker weiter. Dort entscheidet der Richtungs-Knopf.
                    click: (e) => {
                      if (e.originalEvent?.target?.closest?.('.station-label')) return
                      onSelectStation(loc.stations[0].id)
                    }
                  }}
                >
                  <Tooltip
                    permanent
                    interactive
                    direction={LABEL_SIDE[loc.key] || 'right'}
                    offset={LABEL_SIDE[loc.key] === 'left' ? [-12, 0] : [12, 0]}
                    opacity={1}
                    className="station-label"
                  >
                    <div className="station-label-name">{loc.name}</div>
                    <div className="station-label-dirs">
                      {loc.stations.map((s) => (
                        <button
                          key={s.id}
                          type="button"
                          onClick={() => onSelectStation(s.id)}
                          className={`station-label-dir ${
                            s.id === selectedStationId ? 'is-active' : ''
                          }`}
                          title={`${s.name}, Richtung ${s.direction}: Live-Vorhersage öffnen`}
                        >
                          {s.direction}
                          <span className="font-mono">{s.metrics.mlp.r2.toFixed(2)}</span>
                        </button>
                      ))}
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
