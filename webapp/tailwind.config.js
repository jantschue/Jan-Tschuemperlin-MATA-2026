/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        'bg-base': '#0a0a0f',
        'bg-surface': '#13131a',
        'bg-elevated': '#1c1c27',
        border: '#2a2a3a',
        'text-primary': '#f0f0f5',
        'text-muted': '#6b6b80',
        accent: '#4f8ef7',
        'accent-dim': '#1e3a6e',
        success: '#22c55e',
        warning: '#f59e0b',
        danger: '#ef4444'
      },
      fontFamily: {
        mono: ['"DM Mono"', '"Space Mono"', 'ui-monospace', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif']
      },
      borderRadius: {
        DEFAULT: '4px',
        sm: '2px',
        md: '4px'
      }
    }
  },
  plugins: []
}
