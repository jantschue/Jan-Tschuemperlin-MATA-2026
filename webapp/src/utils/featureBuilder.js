/**
 * Wandelt die UI-Eingaben in den Feature-Vektor des v8-Modells um. v8 baut auf
 * v7 auf (26 kantonsspezifische Feiertagsspalten holiday_AG … holiday_ZH statt
 * binärem is_holiday) und ergänzt zusätzlich 26 kantonsspezifische
 * Schulferienspalten (schoolholiday_AG … schoolholiday_ZH); der volle
 * Feature-Satz hat damit 67 Spalten:
 *
 *   Year, Hour_sin/cos, DayOfWeek_sin/cos, Month_sin/cos, DayOfYear_sin/cos,
 *   is_weekend, holiday_AG … holiday_ZH (26), schoolholiday_AG … schoolholiday_ZH
 *   (26), temp, rain_1h, sun_1h, snow_1h,
 *   weather_cat (LabelEncoded: 0=Clear, 1=Cloudy, 2=Night, 3=Rain, 4=Snow)
 *
 * Wichtig: Der Vektor wird NICHT in fester Reihenfolge gebaut, sondern anhand
 * der pro Station exportierten `features`-Liste zusammengesetzt. Damit passt er
 * automatisch auch für Stationen mit Ausnahmen (Sattel R1/R2 enthalten kein
 * `Year`, siehe FEATURE_EXCLUDE in models/mlp_v8.py → 66 statt 67 Features).
 *
 * Feiertage: Die UI kennt einen einzigen "Feiertag"-Schalter. Fällt das
 * eingestellte Datum auf einen Schwyzer Feiertag, werden die 26 holiday_*-
 * Spalten genau so gesetzt, wie das Modell diesen Tag im Training gesehen hat
 * (z. B. Auffahrt: alle Kantone, Josefstag: NW, SZ, TI, UR, VS). Wird der
 * Schalter an einem gewöhnlichen Datum eingeschaltet, gilt das Muster eines
 * landesweiten Feiertags (alle 26 Kantone = 1), weil im Training kein Tag
 * vorkommt, an dem nur holiday_SZ = 1 ist.
 *
 * Schulferien: Der Schalter setzt schoolholiday_SZ = 1, alle übrigen Kantone
 * bleiben 0.
 *
 * Wetter: Einheiten wie im Training (MeteoSchweiz): sun_1h = Sonnenscheindauer
 * in Minuten pro Stunde (0–60), snow_1h = Schneezuwachs in cm pro Stunde.
 * weather_cat folgt der Regel aus scripts/categorize_weather.py: Schnee und
 * Regen gehen vor, sonst gilt zwischen 20 und 6 Uhr immer "Night" (2).
 *
 * Encodings (Periode, Offset) müssen mit der Daten-Engineering-Pipeline
 * (scripts/add_time_features.py) übereinstimmen.
 */

import { findSwissHoliday } from './swissHolidays.js'

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
 *   temp (°C), rain (mm), sun (Minuten 0..60), snow (cm),
 *   weatherCat: 0..4 (LabelEncoder-Index),
 *   isHoliday:       boolean (Feiertag, Kantonsmuster siehe oben),
 *   isSchoolHoliday: boolean (als Schwyzer Schulferien interpretiert)
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
  const isSchoolHoliday = inputs.isSchoolHoliday ? 1 : 0

  // Kantone mit Feiertag: Muster des echten Feiertags, sonst landesweit
  let holidayCantons = []
  if (inputs.isHoliday) {
    const h = findSwissHoliday(inputs.year, inputs.month, day)
    holidayCantons = h ? h.cantons : CANTONS
  }

  // Night-Regel wie in scripts/categorize_weather.py (Snow > Rain > Night)
  const isNightHour = inputs.hour >= 20 || inputs.hour <= 6
  const weatherCat =
    isNightHour && (inputs.weatherCat === 0 || inputs.weatherCat === 1)
      ? 2
      : inputs.weatherCat

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
    weather_cat: weatherCat
  }
  // 26 Kantons-Feiertagsspalten (Muster siehe oben)
  for (const c of CANTONS) map[`holiday_${c}`] = holidayCantons.includes(c) ? 1 : 0
  // 26 Kantons-Schulferienspalten (v8): einziger UI-Schalter -> schoolholiday_SZ
  for (const c of CANTONS) map[`schoolholiday_${c}`] = c === 'SZ' ? isSchoolHoliday : 0

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
// "Night" ist datengetrieben (Tageszeit) und wird nicht in der UI angeboten;
// buildFeatureVector setzt es zwischen 20 und 6 Uhr automatisch.
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
  sun: 40, // Minuten Sonnenschein pro Stunde (0–60)
  snow: 0,
  weatherCat: 0, // Clear
  isHoliday: false,
  isSchoolHoliday: false
}
