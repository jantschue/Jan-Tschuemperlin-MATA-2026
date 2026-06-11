import { useCallback, useEffect, useState } from 'react'

/**
 * Verwaltet das Farbschema (dark / light) der Webapp.
 *
 * Das Theme wird als `data-theme`-Attribut auf <html> gesetzt; die eigentlichen
 * Farben kommen aus den CSS-Custom-Properties in index.css. Der initiale Wert
 * wird bereits vom Inline-Skript in index.html gesetzt (kein Flackern beim
 * Laden), dieser Hook liest ihn nur aus und hält ihn synchron.
 *
 * Beim Umschalten wird kurz die Klasse `theme-transition` auf <html> gesetzt,
 * damit der Wechsel sanft animiert (siehe index.css).
 */

const STORAGE_KEY = 'theme'
const TRANSITION_MS = 420

// Aktuelles Theme aus dem <html>-Attribut lesen (vom Inline-Skript gesetzt)
function readTheme() {
  if (typeof document === 'undefined') return 'dark'
  return document.documentElement.getAttribute('data-theme') === 'light'
    ? 'light'
    : 'dark'
}

export function useTheme() {
  const [theme, setTheme] = useState(readTheme)

  // Theme mit dem <html>-Attribut synchron halten: Ein MutationObserver sorgt
  // dafuer, dass *alle* Komponenten, die diesen Hook nutzen (z.B. die
  // Stationskarte), auf einen Umschalter an anderer Stelle reagieren – nicht
  // nur jene Komponente, die toggleTheme tatsaechlich aufruft.
  useEffect(() => {
    const root = document.documentElement
    const sync = () =>
      setTheme((prev) => {
        const current = readTheme()
        return prev === current ? prev : current
      })
    sync() // Initialwert (z.B. System-Preference) angleichen
    const observer = new MutationObserver(sync)
    observer.observe(root, { attributes: true, attributeFilter: ['data-theme'] })
    return () => observer.disconnect()
  }, [])

  const toggleTheme = useCallback(() => {
    setTheme((prev) => {
      const next = prev === 'dark' ? 'light' : 'dark'
      const root = document.documentElement

      // Sanften Uebergang nur waehrend des Umschaltens aktivieren
      root.classList.add('theme-transition')
      root.setAttribute('data-theme', next)
      try {
        localStorage.setItem(STORAGE_KEY, next)
      } catch (e) {
        /* localStorage nicht verfuegbar – Theme gilt nur fuer diese Sitzung */
      }

      window.clearTimeout(useTheme._t)
      useTheme._t = window.setTimeout(
        () => root.classList.remove('theme-transition'),
        TRANSITION_MS
      )
      return next
    })
  }, [])

  return [theme, toggleTheme]
}
