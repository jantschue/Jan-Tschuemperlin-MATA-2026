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
