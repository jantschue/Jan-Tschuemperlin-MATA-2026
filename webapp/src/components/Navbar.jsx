// Obere Navigationsleiste mit App-Titel, Ansichts-Links und Stations-Dropdown
const VIEWS = [
  { id: 'karte', label: 'Stationskarte' },
  { id: 'vorhersage', label: 'Live-Vorhersage' },
  { id: 'datum', label: 'Datums-Analyse' },
  { id: 'sensitivitaet', label: 'Feature-Sensitivität' },
  { id: 'anomalie', label: 'Anomalie-Explorer' }
]

export default function Navbar({
  view,
  onChangeView,
  stations,
  selectedStationId,
  onChangeStation
}) {
  return (
    <header className="border-b border-[var(--border)] bg-[var(--bg-base)]/85 backdrop-blur supports-[backdrop-filter]:bg-[var(--bg-base)]/70 sticky top-0 z-30">
      <div className="max-w-[1320px] mx-auto h-16 px-6 md:px-12 flex items-center gap-6">
        {/* Wortmarke */}
        <a
          href="#"
          onClick={(e) => {
            e.preventDefault()
            onChangeView('karte')
          }}
          className="flex items-center gap-2.5 group shrink-0"
        >
          <span className="inline-block w-2 h-2 rounded-full bg-[var(--accent)] group-hover:scale-110 transition-transform" />
          <span className="text-[0.92rem] tracking-tightish">
            Jan MATA
            <span className="text-[var(--text-muted)]"> · 2026</span>
          </span>
        </a>

        {/* Desktop-Navigation */}
        <nav className="hidden md:flex items-center gap-1 ml-6">
          {VIEWS.map((v) => {
            const active = view === v.id
            return (
              <button
                key={v.id}
                onClick={() => onChangeView(v.id)}
                className={`relative px-3 py-1.5 text-[0.875rem] rounded-sm transition-colors ${
                  active
                    ? 'text-[var(--text-primary)]'
                    : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'
                }`}
              >
                {v.label}
                {active && (
                  <span className="absolute left-3 right-3 -bottom-[19px] h-px bg-[var(--accent)]" />
                )}
              </button>
            )
          })}
        </nav>

        {/* Mobile-Navigation */}
        <select
          className="md:hidden flex-1"
          value={view}
          onChange={(e) => onChangeView(e.target.value)}
        >
          {VIEWS.map((v) => (
            <option key={v.id} value={v.id}>
              {v.label}
            </option>
          ))}
        </select>

        {/* Stations-Picker */}
        <div className="ml-auto flex items-center gap-2.5 shrink-0">
          <span className="hidden md:inline eyebrow">Station</span>
          <select
            value={selectedStationId || ''}
            onChange={(e) => onChangeStation(e.target.value)}
            className="min-w-[180px]"
          >
            {stations.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name} · {s.direction}
              </option>
            ))}
          </select>
        </div>
      </div>
    </header>
  )
}
