import { useEffect, useState } from 'react'
import Navbar from './components/Navbar.jsx'
import StationsMap from './components/StationsMap.jsx'
import LiveVorhersage from './components/LiveVorhersage.jsx'
import DatumsAnalyse from './components/DatumsAnalyse.jsx'
import FeatureSensitivitaet from './components/FeatureSensitivitaet.jsx'

// Hauptkomponente: globaler Zustand (aktive Ansicht + gewählte Station)
export default function App() {
  const [view, setView] = useState('karte')
  const [stations, setStations] = useState([])
  const [selectedStationId, setSelectedStationId] = useState(null)
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

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center px-6">
        <div className="card p-8 max-w-md text-center">
          <h2 className="text-xl mb-2">Daten konnten nicht geladen werden</h2>
          <p className="text-[var(--text-muted)] text-sm">{error}</p>
        </div>
      </div>
    )
  }

  if (stations.length === 0) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-[var(--text-muted)] text-sm">Stationen werden geladen...</p>
      </div>
    )
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar
        view={view}
        onChangeView={setView}
        stations={stations}
        selectedStationId={selectedStationId}
        onChangeStation={setSelectedStationId}
      />

      <main className="flex-1 px-6 md:px-10 py-8 max-w-[1400px] w-full mx-auto">
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
          <DatumsAnalyse station={selectedStation} />
        )}
        {view === 'sensitivitaet' && selectedStation && (
          <FeatureSensitivitaet station={selectedStation} />
        )}
      </main>

      <footer className="px-6 md:px-10 py-4 border-t border-[var(--border)] text-xs text-[var(--text-muted)] flex justify-between">
        <span>Maturaarbeit 2026 · ASTRA-Verkehrsdaten Kanton Schwyz</span>
        <span className="font-mono">v0.1</span>
      </footer>
    </div>
  )
}
