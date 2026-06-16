/**
 * Modell-Vorwärtsdurchläufe für MLP und Lineare Regression. Die Implementierung
 * spiegelt exakt die in models/mlp_v7.py und models/linear_regression_v7.py
 * definierte Architektur wider:
 *
 * MLP (mlp.py):
 *   Pro Hidden-Block: Linear -> BatchNorm1d -> ReLU -> (Dropout)
 *   Output: Linear (1 Neuron, keine Aktivierung)
 *   Dropout fällt im Eval-Modus weg und wird daher nicht im JSON exportiert.
 *
 * LinearRegressionModel (linear_regression.py):
 *   y = W·x + b mit einem einzigen nn.Linear(input_dim, 1)
 *
 * In beiden Fällen wird sowohl die Eingabe als auch das Ziel mit einem
 * StandardScaler skaliert. Beide Skalierungen werden mitexportiert.
 *
 * MLP-JSON-Schema:
 *   {
 *     features:     [...],
 *     layers: [
 *       { type: "linear",    weight: [[...]], bias: [...] },
 *       { type: "batchnorm", weight: [...],   bias: [...],
 *                            running_mean: [...], running_var: [...], eps: 1e-5 },
 *       { type: "relu" },
 *       ...
 *     ],
 *     activation:    "relu",
 *     input_scaler:  { mean: [...], std: [...] },
 *     output_scaler: { mean: [x],   std: [y]   }
 *   }
 */

// Eingabe-Vektor normalisieren: (x - mean) / std
function scaleInput(x, scaler) {
  const out = new Array(x.length)
  for (let i = 0; i < x.length; i++) {
    const std = scaler.std[i] || 1
    out[i] = (x[i] - scaler.mean[i]) / std
  }
  return out
}

// Ausgabe-Skalierung umkehren und auf >= 0 begrenzen
function unscaleOutput(y, scaler) {
  const mean = Array.isArray(scaler.mean) ? scaler.mean[0] : scaler.mean
  const std = Array.isArray(scaler.std) ? scaler.std[0] : scaler.std
  return Math.max(0, Math.round(y * std + mean))
}

// Lineare Schicht: z = W·x + b  (W mit Shape [out_dim, in_dim])
function linearLayer(input, weight, bias) {
  const outDim = weight.length
  const out = new Array(outDim)
  for (let i = 0; i < outDim; i++) {
    let s = bias[i] || 0
    const row = weight[i]
    for (let j = 0; j < row.length; j++) s += row[j] * input[j]
    out[i] = s
  }
  return out
}

// BatchNorm1d im Eval-Modus:
// y_i = gamma_i * (x_i - running_mean_i) / sqrt(running_var_i + eps) + beta_i
function batchNormLayer(input, gamma, beta, runningMean, runningVar, eps) {
  const out = new Array(input.length)
  for (let i = 0; i < input.length; i++) {
    const inv = 1 / Math.sqrt(runningVar[i] + eps)
    out[i] = gamma[i] * (input[i] - runningMean[i]) * inv + beta[i]
  }
  return out
}

function reluVector(v) {
  const out = new Array(v.length)
  for (let i = 0; i < v.length; i++) out[i] = v[i] > 0 ? v[i] : 0
  return out
}

/**
 * MLP-Vorwärtsdurchlauf gemäss exportiertem Schicht-Stapel.
 * @param {number[]} featureVector - Rohe (unskalierte) Feature-Werte
 * @param {object}   weights       - Geladenes MLP-Weights-JSON
 * @returns {number} Vorhergesagte Fahrzeuganzahl (>= 0, ganzzahlig)
 */
export function mlpForward(featureVector, weights) {
  if (!weights || !weights.layers) return 0

  let x = scaleInput(featureVector, weights.input_scaler)

  for (const layer of weights.layers) {
    switch (layer.type) {
      case 'linear':
        x = linearLayer(x, layer.weight, layer.bias)
        break
      case 'batchnorm':
        x = batchNormLayer(
          x,
          layer.weight,
          layer.bias,
          layer.running_mean,
          layer.running_var,
          layer.eps ?? 1e-5
        )
        break
      case 'relu':
        x = reluVector(x)
        break
      // Dropout & Unbekanntes: identity (kein Effekt in Eval)
      default:
        break
    }
  }

  // Output-Schicht liefert genau einen Wert
  return unscaleOutput(x[0], weights.output_scaler)
}

/**
 * Lineare Regression: y = w · x + b, beides skaliert.
 */
export function linearForward(featureVector, weights) {
  if (!weights || !weights.weight) return 0

  const x = scaleInput(featureVector, weights.input_scaler)
  let y = weights.bias || 0
  for (let i = 0; i < x.length; i++) y += weights.weight[i] * x[i]
  return unscaleOutput(y, weights.output_scaler)
}
