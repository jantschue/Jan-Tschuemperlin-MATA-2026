import { useEffect, useState } from 'react'

/**
 * Lädt und cached MLP- und Linear-Weights pro Station-ID.
 * Während dem Laden gibt der Hook { loading: true } zurück.
 * Bei Fehlschlag (Datei fehlt) wird { error: 'missing' } gesetzt.
 */
const cache = new Map()

async function fetchWeights(stationId) {
  if (cache.has(stationId)) return cache.get(stationId)

  const mlpUrl = `/data/weights/${stationId}_mlp.json`
  const lrUrl = `/data/weights/${stationId}_linear.json`

  const [mlpRes, lrRes] = await Promise.all([fetch(mlpUrl), fetch(lrUrl)])
  if (!mlpRes.ok || !lrRes.ok) {
    const err = new Error(`Weights für Station ${stationId} fehlen`)
    err.code = 'missing'
    throw err
  }
  const [mlp, linear] = await Promise.all([mlpRes.json(), lrRes.json()])
  const value = { mlp, linear }
  cache.set(stationId, value)
  return value
}

export function useModelWeights(stationId) {
  const [state, setState] = useState({ loading: true })

  useEffect(() => {
    if (!stationId) return
    let cancelled = false
    setState({ loading: true })
    fetchWeights(stationId)
      .then((weights) => {
        if (!cancelled) setState({ loading: false, weights })
      })
      .catch((err) => {
        if (!cancelled) setState({ loading: false, error: err.code || 'error' })
      })
    return () => {
      cancelled = true
    }
  }, [stationId])

  return state
}

/**
 * Lädt die vorberechneten täglichen Ergebnisse einer Station.
 */
const dailyCache = new Map()

export function useDailyResults(stationId) {
  const [state, setState] = useState({ loading: true })

  useEffect(() => {
    if (!stationId) return
    let cancelled = false
    setState({ loading: true })

    const load = async () => {
      if (dailyCache.has(stationId)) {
        if (!cancelled) setState({ loading: false, data: dailyCache.get(stationId) })
        return
      }
      try {
        const res = await fetch(`/data/results/${stationId}_daily.json`)
        if (!res.ok) throw new Error('missing')
        const data = await res.json()
        dailyCache.set(stationId, data)
        if (!cancelled) setState({ loading: false, data })
      } catch (err) {
        if (!cancelled) setState({ loading: false, error: 'missing' })
      }
    }
    load()
    return () => {
      cancelled = true
    }
  }, [stationId])

  return state
}

/**
 * Lädt die rohen Feiertags-Feature-Vektoren einer Station (aus
 * export_weights.py). Jeder Eintrag: { datetime, f: [16 Roh-Features] }.
 * Wird für das Counterfactual gebraucht (is_holiday auf 0 setzen).
 */
const holidayCache = new Map()

export function useHolidayFeatures(stationId) {
  const [state, setState] = useState({ loading: true })

  useEffect(() => {
    if (!stationId) return
    let cancelled = false
    setState({ loading: true })

    const load = async () => {
      if (holidayCache.has(stationId)) {
        if (!cancelled) setState({ loading: false, data: holidayCache.get(stationId) })
        return
      }
      try {
        const res = await fetch(`/data/features/${stationId}_holidays.json`)
        if (!res.ok) throw new Error('missing')
        const data = await res.json()
        holidayCache.set(stationId, data)
        if (!cancelled) setState({ loading: false, data })
      } catch (err) {
        if (!cancelled) setState({ loading: false, error: 'missing' })
      }
    }
    load()
    return () => {
      cancelled = true
    }
  }, [stationId])

  return state
}

/**
 * Lädt die täglichen Ergebnisse ALLER Stationen (für stationsübergreifende
 * Aggregationen). Lädt nur, wenn `enabled` true ist, und cached pro Station.
 * Liefert ein flaches Array mit ergänztem stationId-Feld.
 */
export function useAllDailyResults(stations, enabled) {
  const [state, setState] = useState({ loading: false, data: null })

  useEffect(() => {
    if (!enabled || !stations?.length) return
    let cancelled = false
    setState({ loading: true, data: null })

    Promise.all(
      stations.map(async (s) => {
        if (dailyCache.has(s.id)) return { id: s.id, rows: dailyCache.get(s.id) }
        try {
          const res = await fetch(`/data/results/${s.id}_daily.json`)
          if (!res.ok) return { id: s.id, rows: [] }
          const rows = await res.json()
          dailyCache.set(s.id, rows)
          return { id: s.id, rows }
        } catch {
          return { id: s.id, rows: [] }
        }
      })
    ).then((chunks) => {
      if (cancelled) return
      const data = chunks.flatMap((c) =>
        c.rows.map((r) => ({ ...r, stationId: c.id }))
      )
      setState({ loading: false, data })
    })

    return () => {
      cancelled = true
    }
  }, [stations, enabled])

  return state
}
