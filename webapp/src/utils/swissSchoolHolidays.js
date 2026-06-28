/**
 * Schulferien des Kantons Schwyz.
 *
 * Anders als die gesetzlichen Feiertage (formelbasiert, siehe swissHolidays.js)
 * folgen die Schulferien keinem berechenbaren Muster – sie stammen aus der
 * kantonalen Schulferien-CSV. export_weights.py exportiert daraus den gemeinsamen
 * Kalender `webapp/public/data/features/schoolholidays_sz.json` als Liste
 * benannter Perioden `[{ name, start, end }]` (Datumsstrings YYYY-MM-DD).
 *
 * Dieses Modul wandelt diesen rohen Kalender in die für die Schulferien-Analyse
 * benötigten Strukturen um (Datums-Zuordnung, Datumsmengen, Reihenfolge).
 */

// Zweistellige Zahl mit führender Null
function pad2(n) {
  return String(n).padStart(2, '0')
}

// Iteriert alle Kalendertage von start bis end (inkl.) als YYYY-MM-DD-Strings
function eachDate(startStr, endStr) {
  const out = []
  const [ys, ms, ds] = startStr.split('-').map(Number)
  const [ye, me, de] = endStr.split('-').map(Number)
  const cur = new Date(ys, ms - 1, ds)
  const end = new Date(ye, me - 1, de)
  while (cur <= end) {
    out.push(`${cur.getFullYear()}-${pad2(cur.getMonth() + 1)}-${pad2(cur.getDate())}`)
    cur.setDate(cur.getDate() + 1)
  }
  return out
}

/**
 * Wandelt den rohen Kalender in einzelne Ferien-Vorkommen um.
 * @param {Array<{name,start,end}>} schedule
 * @returns {Array<{id,name,year,start,end,dates:string[]}>}
 *   id eindeutig pro Vorkommen, year = Startjahr (Weihnachtsferien über den
 *   Jahreswechsel zählen zum Startjahr).
 */
export function schoolPeriods(schedule) {
  if (!schedule) return []
  return schedule.map((p) => ({
    id: `${p.name}__${p.start}`,
    name: p.name,
    year: Number(p.start.slice(0, 4)),
    start: p.start,
    end: p.end,
    dates: eachDate(p.start, p.end)
  }))
}

/**
 * Map dateStr -> Vorkommen-Metadaten ({ id, name, year }) für schnelle Zuordnung.
 */
export function schoolDateMap(schedule) {
  const map = new Map()
  for (const p of schoolPeriods(schedule)) {
    for (const d of p.dates) map.set(d, { id: p.id, name: p.name, year: p.year })
  }
  return map
}

/**
 * Set aller Schulferien-Datumsstrings – um Schulferien aus der Wochentags-
 * Referenz auszuschliessen (Baseline = nur normale Schulwochen).
 */
export function schoolDateSet(schedule) {
  const set = new Set()
  for (const p of schoolPeriods(schedule)) {
    for (const d of p.dates) set.add(d)
  }
  return set
}

// Reihenfolge der Schulferien im Schuljahr (für Ranking-Achsen)
export const SCHOOL_ORDER = [
  'Sportferien', 'Frühlingsferien', 'Sommerferien', 'Herbstferien', 'Weihnachtsferien'
]
