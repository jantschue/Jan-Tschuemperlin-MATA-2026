/**
 * Schweizer Feiertage für den Kanton Schwyz.
 *
 * getSwissHolidays(year) liefert alle verkehrsrelevanten Feiertage eines Jahres
 * mit Datum (YYYY-MM-DD), Name und Kategorie. Das Osterdatum wird mit der
 * Gauss-Osterformel (Butcher-Algorithmus) berechnet, ohne externe API.
 *
 * Die identische Datumslogik liegt serverseitig in export_weights.py
 * (swiss_holiday_dates), damit die exportierten Feiertags-Feature-Vektoren und
 * die hier angezeigten Feiertage exakt übereinstimmen.
 *
 * Kategorien:
 *   fix        – gesetzlich, festes Kalenderdatum (z.B. Neujahr, Bundesfeier)
 *   beweglich  – ostergebunden (z.B. Karfreitag, Auffahrt)
 *   kirchlich  – fixes kirchliches Datum (z.B. Allerheiligen, Weihnachten)
 *
 * Berchtoldstag (2.1.) ist im Kanton Schwyz nicht gesetzlich, aber
 * verkehrsrelevant und wird mit optional=true markiert.
 */

// Zweistellige Zahl mit führender Null
function pad2(n) {
  return String(n).padStart(2, '0')
}

// Datum (Jahr/Monat/Tag) als lokaler YYYY-MM-DD-String, ohne Zeitzonen-Drift
function isoLocal(year, month, day) {
  return `${year}-${pad2(month)}-${pad2(day)}`
}

// Ostersonntag nach Butcher-Algorithmus
export function easterSunday(year) {
  const a = year % 19
  const b = Math.floor(year / 100)
  const c = year % 100
  const d = Math.floor(b / 4)
  const e = b % 4
  const f = Math.floor((b + 8) / 25)
  const g = Math.floor((b - f + 1) / 3)
  const h = (19 * a + b - d - g + 15) % 30
  const i = Math.floor(c / 4)
  const k = c % 4
  const l = (32 + 2 * e + 2 * i - h - k) % 7
  const m = Math.floor((a + 11 * h + 22 * l) / 451)
  const month = Math.floor((h + l - 7 * m + 114) / 31)
  const day = ((h + l - 7 * m + 114) % 31) + 1
  return new Date(year, month - 1, day)
}

// Tage zu einem Date addieren (lokal, ohne Zeitkomponente)
function addDays(date, days) {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate() + days)
}

/**
 * Alle Schwyzer Feiertage eines Jahres.
 * @param {number} year
 * @returns {Array<{date:string, name:string, category:string, optional?:boolean}>}
 */
export function getSwissHolidays(year) {
  const easter = easterSunday(year)
  const mov = (offset) => {
    const d = addDays(easter, offset)
    return isoLocal(d.getFullYear(), d.getMonth() + 1, d.getDate())
  }

  return [
    { date: isoLocal(year, 1, 1),  name: 'Neujahr',            category: 'fix' },
    { date: isoLocal(year, 1, 2),  name: 'Berchtoldstag',      category: 'fix',       optional: true },
    { date: mov(-2),               name: 'Karfreitag',         category: 'beweglich' },
    { date: mov(1),                name: 'Ostermontag',        category: 'beweglich' },
    { date: mov(39),               name: 'Auffahrt',           category: 'beweglich' },
    { date: mov(50),               name: 'Pfingstmontag',      category: 'beweglich' },
    { date: mov(60),               name: 'Fronleichnam',       category: 'beweglich' },
    { date: isoLocal(year, 8, 1),  name: 'Bundesfeier',        category: 'fix' },
    { date: isoLocal(year, 8, 15), name: 'Mariä Himmelfahrt',  category: 'kirchlich' },
    { date: isoLocal(year, 11, 1), name: 'Allerheiligen',      category: 'kirchlich' },
    { date: isoLocal(year, 12, 8), name: 'Mariä Empfängnis',   category: 'kirchlich' },
    { date: isoLocal(year, 12, 25), name: 'Weihnachten',       category: 'kirchlich' },
    { date: isoLocal(year, 12, 26), name: 'Stephanstag',       category: 'kirchlich' },
  ]
}

/**
 * Liefert für mehrere Jahre eine Map dateStr -> Feiertags-Metadaten.
 * @param {number[]} years
 * @param {object} opts
 * @param {boolean} opts.includeOptional - Berchtoldstag mit einbeziehen
 */
export function holidayMap(years, { includeOptional = false } = {}) {
  const map = new Map()
  for (const y of years) {
    for (const h of getSwissHolidays(y)) {
      if (h.optional && !includeOptional) continue
      map.set(h.date, h)
    }
  }
  return map
}

/**
 * Set aller Feiertags-Datumsstrings (inkl. Berchtoldstag) über mehrere Jahre.
 * Wird genutzt, um Feiertage aus der Wochentags-Referenz auszuschliessen,
 * damit die Baseline nur "normale" Tage enthält.
 */
export function holidayDateSet(years) {
  const set = new Set()
  for (const y of years) {
    for (const h of getSwissHolidays(y)) set.add(h.date)
  }
  return set
}

// Feste Reihenfolge der Feiertage im Kalenderjahr (für Ranking-Achsen)
export const HOLIDAY_ORDER = [
  'Neujahr', 'Berchtoldstag', 'Karfreitag', 'Ostermontag', 'Auffahrt',
  'Pfingstmontag', 'Fronleichnam', 'Bundesfeier', 'Mariä Himmelfahrt',
  'Allerheiligen', 'Mariä Empfängnis', 'Weihnachten', 'Stephanstag',
]
