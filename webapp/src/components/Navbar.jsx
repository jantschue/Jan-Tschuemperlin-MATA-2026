// Obere Navigationsleiste mit App-Titel, Ansichts-Links und Stations-Dropdown
const VIEWS = [
  { id: 'karte', label: 'Stationskarte' },
  { id: 'vorhersage', label: 'Live-Vorhersage' },
  { id: 'datum', label: 'Datums-Analyse' },
  { id: 'sensitivitaet', label: 'Feature-Sensitivität' }
]

export default function Navbar({
  view,
  onChangeView,
  stations,
  selectedStationId,
  onChangeStation
}) {
  return (
    <header className="h-14 border-b border-[var(--border)] bg-[var(--bg-base)] flex items-center px-6 md:px-10 sticky top-0 z-30">
      <div className="font-mono text-sm tracking-wide text-[var(--text-primary)]">
        Jan-Tschümperlin<span className="text-[var(--accent)]">-MATA-2026</span>
      </div>

      <nav className="hidden md:flex ml-12 gap-8 flex-1">
        {VIEWS.map((v) => {
          const active = view === v.id
          return (
            <button
              key={v.id}
              onClick={() => onChangeView(v.id)}
              className={`relative text-sm py-1 transition-colors ${
                active ? 'text-[var(--text-primary)]' : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'
              }`}
            >
              {v.label}
              {active && (
                <span className="absolute left-0 right-0 -bottom-[19px] h-[2px] bg-[var(--accent)]" />
              )}
            </button>
          )
        })}
      </nav>

      {/* Mobile-Navigation: Dropdown */}
      <select
        className="md:hidden ml-4 flex-1"
        value={view}
        onChange={(e) => onChangeView(e.target.value)}
      >
        {VIEWS.map((v) => (
          <option key={v.id} value={v.id}>
            {v.label}
          </option>
        ))}
      </select>

      <div className="ml-auto md:ml-4 flex items-center gap-3">
        <span className="hidden md:inline text-xs uppercase tracking-wider text-[var(--text-muted)]">
          Station
        </span>
        <select
          value={selectedStationId || ''}
          onChange={(e) => onChangeStation(e.target.value)}
        >
          {stations.map((s) => (
            <option key={s.id} value={s.id}>
              {s.name} · {s.direction}
            </option>
          ))}
        </select>
      </div>
    </header>
  )
}
