import { useTheme } from '../hooks/useTheme.js'

// Obere Navigationsleiste mit App-Titel, Ansichts-Links und Stations-Dropdown
const VIEWS = [
  { id: 'karte', label: 'Stationskarte' },
  { id: 'vorhersage', label: 'Live-Vorhersage' },
  { id: 'datum', label: 'Datums-Analyse' },
  { id: 'feiertage', label: 'Feiertage' },
  { id: 'schulferien', label: 'Schulferien' },
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

        {/* Rechter Bereich: Stations-Picker + Theme-Schalter */}
        <div className="ml-auto flex items-center gap-3 shrink-0">
          <div className="flex items-center gap-2.5">
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
          <ThemeToggle />
        </div>
      </div>
    </header>
  )
}

// Schalter zwischen Dark und Light Mode mit sanftem Icon-Wechsel
function ThemeToggle() {
  const [theme, toggleTheme] = useTheme()
  const isDark = theme === 'dark'

  return (
    <button
      type="button"
      onClick={toggleTheme}
      aria-label={isDark ? 'Zum hellen Modus wechseln' : 'Zum dunklen Modus wechseln'}
      title={isDark ? 'Heller Modus' : 'Dunkler Modus'}
      className="relative grid place-items-center w-9 h-9 rounded-sm border border-[var(--border)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:border-[var(--border-strong)] hover:bg-[var(--bg-elevated)] transition-colors shrink-0"
    >
      {/* Sonne – sichtbar im Dark Mode (Klick fuehrt zu Hell) */}
      <SunIcon
        className={`absolute inset-0 m-auto transition-all duration-300 ${
          isDark ? 'opacity-100 rotate-0 scale-100' : 'opacity-0 -rotate-90 scale-50'
        }`}
      />
      {/* Mond – sichtbar im Light Mode (Klick fuehrt zu Dunkel) */}
      <MoonIcon
        className={`absolute inset-0 m-auto transition-all duration-300 ${
          isDark ? 'opacity-0 rotate-90 scale-50' : 'opacity-100 rotate-0 scale-100'
        }`}
      />
    </button>
  )
}

function SunIcon({ className = '' }) {
  return (
    <svg
      className={className}
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
    </svg>
  )
}

function MoonIcon({ className = '' }) {
  return (
    <svg
      className={className}
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
    </svg>
  )
}
