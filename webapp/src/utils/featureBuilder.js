/**
 * Wandelt die UI-Eingaben in den Feature-Vektor des v7-Modells um. v7 ersetzt
 * das binäre is_holiday durch 26 kantonsspezifische Feiertagsspalten
 * (holiday_AG … holiday_ZH); der volle Feature-Satz hat damit 41 Spalten:
 *
 *   Year, Hour_sin/cos, DayOfWeek_sin/cos, Month_sin/cos, DayOfYear_sin/cos,
 *   is_weekend, holiday_AG … holiday_ZH (26), temp, rain_1h, sun_1h, snow_1h,
 *   weather_cat (LabelEncoded: 0=Clear, 1=Cloudy, 2=Night, 3=Rain, 4=Snow)
 *
 * Wichtig: Der Vektor wird NICHT in fester Reihenfolge gebaut, sondern anhand
 * der pro Station exportierten `features`-Liste zusammengesetzt. Damit passt er
 * automatisch auch für Stationen mit Ausnahmen (Sattel R1/R2 enthalten kein
 * `Year`, siehe FEATURE_EXCLUDE in models/mlp_v7.py → 40 statt 41 Features).
 *
 * Feiertage: Die Live-UI kennt nur einen einzigen "Feiertag"-Schalter. Da das
 * Projektgebiet im Kanton Schwyz liegt, wird dieser als Schwyzer Feiertag
 * interpretiert (holiday_SZ = 1, alle übrigen Kantone 0).
 *
 * Encodings (Periode, Offset) müssen mit der Daten-Engineering-Pipeline
 * (scripts/add_time_features.py) übereinstimmen.
 */

// 26 Schweizer Kantonskürzel (alphabetisch, wie in scripts/generate_holidays.py)
const CANTONS = [
  'AG', 'AI', 'AR', 'BE', 'BL', 'BS', 'FR', 'GE', 'GL', 'GR',
  'JU', 'LU', 'NE', 'NW', 'OW', 'SG', 'SH', 'SO', 'SZ', 'TG',
  'TI', 'UR', 'VD', 'VS', 'ZG', 'ZH'
]

// Tag-im-Jahr aus Monat (1..12), Tag (1..31)
function dayOfYear(month, day) {
  const daysPerMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
  let total = day
  for (let i = 0; i < month - 1; i++) total += daysPerMonth[i]
  return total
}

// Sin/Cos-Kodierung mit Periode (Wert nicht verschieben – passt zur
// Python-Pipeline, die z. B. Month=1 und DayOfYear=1 als ersten Wert nutzt)
function cyclic(value, period) {
  const angle = (2 * Math.PI * value) / period
  return [Math.sin(angle), Math.cos(angle)]
}

/**
 * @param {object} inputs - {
 *   year:       2015..2025,
 *   hour:       0..23,
 *   dayOfWeek:  0..6 (Montag = 0),
 *   month:      1..12,
 *   day:        1..31 (Default 15),
 *   temp, rain, sun, snow,
 *   weatherCat: 0..4 (LabelEncoder-Index),
 *   isHoliday:  boolean (als Schwyzer Feiertag interpretiert)
 * }
 * @param {string[]} featureNames - die pro Station exportierte `features`-Liste
 *   (aus dem Weights-JSON). Bestimmt Reihenfolge und Umfang des Vektors.
 * @returns {number[]} Feature-Vektor in der Reihenfolge von featureNames
 */
export function buildFeatureVector(inputs, featureNames) {
  const day = inputs.day || 15
  const [hourSin, hourCos]   = cyclic(inputs.hour, 24)
  const [dowSin,  dowCos]    = cyclic(inputs.dayOfWeek, 7)
  const [monthSin, monthCos] = cyclic(inputs.month, 12)
  const [doySin,  doyCos]    = cyclic(dayOfYear(inputs.month, day), 365)

  const isWeekend = inputs.dayOfWeek >= 5 ? 1 : 0
  const isHoliday = inputs.isHoliday ? 1 : 0

  // Benannte Feature-Map (Schlüssel = exakte Spaltennamen der Python-Pipeline)
  const map = {
    Year: inputs.year,
    Hour_sin: hourSin, Hour_cos: hourCos,
    DayOfWeek_sin: dowSin, DayOfWeek_cos: dowCos,
    Month_sin: monthSin, Month_cos: monthCos,
    DayOfYear_sin: doySin, DayOfYear_cos: doyCos,
    is_weekend: isWeekend,
    temp: inputs.temp,
    rain_1h: inputs.rain,
    sun_1h: inputs.sun,
    snow_1h: inputs.snow,
    weather_cat: inputs.weatherCat
  }
  // 26 Kantons-Feiertagsspalten: einziger UI-Schalter -> holiday_SZ
  for (const c of CANTONS) map[`holiday_${c}`] = c === 'SZ' ? isHoliday : 0

  // Vektor in der Reihenfolge der modellspezifischen Feature-Liste aufbauen.
  // Unbekannte/fehlende Namen (sollte nicht vorkommen) werden zu 0.
  const names = featureNames && featureNames.length ? featureNames : Object.keys(map)
  return names.map((n) => (n in map ? map[n] : 0))
}

export const WEEKDAYS = [
  'Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag', 'Sonntag'
]

export const MONTHS = [
  'Januar', 'Februar', 'März', 'April', 'Mai', 'Juni',
  'Juli',   'August',  'September', 'Oktober', 'November', 'Dezember'
]

// LabelEncoder-Reihenfolge entspricht alphabetischer Sortierung der Klassen
// im Trainings-Datensatz (Clear, Cloudy, Night, Rain, Snow).
// "Night" ist datengetrieben (Tageszeit) und wird nicht in der UI angeboten.
export const WEATHER_OPTIONS = [
  { id: 'clear',  label: 'Klar',     cat: 0 },
  { id: 'cloudy', label: 'Bewölkt',  cat: 1 },
  { id: 'rain',   label: 'Regen',    cat: 3 },
  { id: 'snow',   label: 'Schnee',   cat: 4 }
]

// Standard-Eingaben für Formulare und Sensitivitätsanalyse
export const DEFAULT_INPUTS = {
  year: 2024,
  hour: 8,
  dayOfWeek: 1, // Dienstag
  month: 6,
  day: 15,
  temp: 18,
  rain: 0,
  sun: 0.7,
  snow: 0,
  weatherCat: 0, // Clear
  isHoliday: false
}
