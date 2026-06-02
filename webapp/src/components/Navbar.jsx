// Obere Navigationsleiste mit App-Titel, Ansichts-Links und Stations-Dropdown
const VIEWS = [
  { id: 'karte', label: 'Stationskarte' },
  { id: 'vorhersage', label: 'Live-Vorhersage' },
  { id: 'datum', label: 'Datums-Analyse' },
  { id: 'feiertage', label: 'Feiertage' },
  { id: 'sensitivitaet', label: 'Feature-Sensitivität' },
  { id: 'anomalie', label: 'Ausreisseranalyse' }
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
      <div className="max-w-[1320px] mx-auto h-16 px-4 md:px-6 lg:px-8 flex items-center gap-4">
        {/* Wortmarke */}
        <a
          href="#"
          onClick={(e) => {
            e.preventDefault()
            onChangeView('karte')
          }}
          className="flex items-center gap-2.5 group shrink-0"
        >
          <span className="text-[0.92rem] tracking-tightish">
            Jan MATA
            <span className="text-[var(--text-muted)]"> · 2026</span>
          </span>
        </a>

        {/* Desktop-Navigation */}
        <nav className="hidden md:flex items-center gap-0">
          {VIEWS.map((v) => {
            const active = view === v.id
            return (
              <button
                key={v.id}
                onClick={() => onChangeView(v.id)}
                className={`relative px-2 py-1.5 text-[0.78rem] rounded-sm transition-colors whitespace-nowrap ${
                  active
                    ? 'text-[var(--text-primary)]'
                    : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'
                }`}
              >
                {v.label}
                {active && (
                  <span className="absolute left-0 right-0 -bottom-[19px] h-px bg-[var(--accent)]" />
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
          <span className="hidden xl:inline eyebrow">Station</span>
          <select
            value={selectedStationId || ''}
            onChange={(e) => onChangeStation(e.target.value)}
            className="w-[152px] xl:min-w-[180px]"
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
