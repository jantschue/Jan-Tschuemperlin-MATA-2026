import { useEffect, useState } from 'react'
import Navbar from './components/Navbar.jsx'
import StationsMap from './components/StationsMap.jsx'
import LiveVorhersage from './components/LiveVorhersage.jsx'
import DatumsAnalyse from './components/DatumsAnalyse.jsx'
import FeatureSensitivitaet from './components/FeatureSensitivitaet.jsx'
import FeiertagsAnalyse from './components/FeiertagsAnalyse.jsx'
import AnomalieExplorer from './components/AnomalieExplorer.jsx'

// Hauptkomponente: globaler Zustand (aktive Ansicht + gewählte Station)
export default function App() {
  const [view, setView] = useState('karte')
  const [stations, setStations] = useState([])
  const [selectedStationId, setSelectedStationId] = useState(null)
  const [datumInitialDate, setDatumInitialDate] = useState('')
  const [error, setError] = useState(null)

  // Stations-Liste einmalig laden
  useEffect(() => {
    fetch('/data/stations.json')
      .then((res) => {
        if (!res.ok) throw new Error('stations.json fehlt')
        return res.json()
      })
      .then((data) => {
        setStations(data)
        if (data.length > 0) setSelectedStationId(data[0].id)
      })
      .catch((err) => setError(err.message))
  }, [])

  const selectedStation =
    stations.find((s) => s.id === selectedStationId) || null

  // Wechselt zur Live-Vorhersage und setzt eine neue Station
  const goToPrediction = (stationId) => {
    setSelectedStationId(stationId)
    setView('vorhersage')
  }

  // Wechselt zur Datums-Analyse mit vorausgewähltem Datum (aus Anomalie-Explorer)
  const goToDatum = (stationId, dateStr) => {
    setSelectedStationId(stationId)
    setDatumInitialDate(dateStr)
    setView('datum')
  }

  if (error) {
    return (
      <div className="min-h-[100dvh] flex items-center justify-center px-6">
        <div className="card p-10 max-w-md text-center">
          <div className="eyebrow mb-3">Ladefehler</div>
          <h2 className="mb-2">Daten konnten nicht geladen werden</h2>
          <p className="text-[var(--text-muted)] text-sm">{error}</p>
        </div>
      </div>
    )
  }

  if (stations.length === 0) {
    return (
      <div className="min-h-[100dvh] flex items-center justify-center">
        <div className="flex items-center gap-3 text-[var(--text-muted)] text-sm">
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-[var(--accent)] animate-pulse" />
          Stationen werden geladen
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-[100dvh] flex flex-col">
      <Navbar
        view={view}
        onChangeView={setView}
        stations={stations}
        selectedStationId={selectedStationId}
        onChangeStation={setSelectedStationId}
      />

      <main className="flex-1 px-6 md:px-12 py-10 md:py-14 max-w-[1320px] w-full mx-auto">
        {view === 'karte' && (
          <StationsMap
            stations={stations}
            selectedStationId={selectedStationId}
            onSelectStation={goToPrediction}
          />
        )}
        {view === 'vorhersage' && selectedStation && (
          <LiveVorhersage
            station={selectedStation}
            onGoToMap={() => setView('karte')}
          />
        )}
        {view === 'datum' && selectedStation && (
          <DatumsAnalyse station={selectedStation} initialDate={datumInitialDate} />
        )}
        {view === 'feiertage' && selectedStation && (
          <FeiertagsAnalyse station={selectedStation} stations={stations} />
        )}
        {view === 'sensitivitaet' && selectedStation && (
          <FeatureSensitivitaet station={selectedStation} />
        )}
        {view === 'anomalie' && (
          <AnomalieExplorer stations={stations} onGoToDatum={goToDatum} />
        )}
      </main>

      <footer className="px-6 md:px-12 py-6 border-t border-[var(--border)] text-xs text-[var(--text-muted)] flex flex-col md:flex-row gap-2 md:gap-0 justify-between max-w-[1320px] w-full mx-auto">
        <span>
          Maturaarbeit 2026 · ASTRA-Verkehrsdaten Kanton Schwyz
        </span>
        <span className="font-mono text-[var(--text-faint)]">v0.1</span>
      </footer>
    </div>
  )
}
