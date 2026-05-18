/**
 * Wandelt die UI-Eingaben in den 16-dimensionalen Feature-Vektor um, der
 * exakt der FEATURES-Liste aus models/mlp.py entspricht:
 *
 *  0: Year             (Kalenderjahr 2015..2025)
 *  1: Hour_sin         (zyklische Stunde, Periode 24)
 *  2: Hour_cos
 *  3: DayOfWeek_sin    (Periode 7, Montag = 0)
 *  4: DayOfWeek_cos
 *  5: Month_sin        (Periode 12, Januar = 1)
 *  6: Month_cos
 *  7: DayOfYear_sin    (Periode 365, 1. Januar = 1)
 *  8: DayOfYear_cos
 *  9: is_weekend
 * 10: is_holiday
 * 11: temp
 * 12: rain_1h
 * 13: sun_1h
 * 14: snow_1h
 * 15: weather_cat      (LabelEncoded: 0=Clear, 1=Cloudy, 2=Night, 3=Rain, 4=Snow)
 *
 * Wichtig: Encodings (Periode, Offset) müssen mit der Daten-Engineering-
 * Pipeline (scripts/add_time_features.py) übereinstimmen.
 */

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
 *   isHoliday:  boolean
 * }
 * @returns {number[]} 16-dimensionaler Feature-Vektor
 */
export function buildFeatureVector(inputs) {
  const day = inputs.day || 15
  const [hourSin, hourCos]   = cyclic(inputs.hour, 24)
  const [dowSin,  dowCos]    = cyclic(inputs.dayOfWeek, 7)
  const [monthSin, monthCos] = cyclic(inputs.month, 12)
  const [doySin,  doyCos]    = cyclic(dayOfYear(inputs.month, day), 365)

  const isWeekend = inputs.dayOfWeek >= 5 ? 1 : 0
  const isHoliday = inputs.isHoliday ? 1 : 0

  return [
    inputs.year,
    hourSin, hourCos,
    dowSin,  dowCos,
    monthSin, monthCos,
    doySin,  doyCos,
    isWeekend,
    isHoliday,
    inputs.temp,
    inputs.rain,
    inputs.sun,
    inputs.snow,
    inputs.weatherCat
  ]
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
