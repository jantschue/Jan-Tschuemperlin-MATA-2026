/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        'bg-base': '#0c0c0d',
        'bg-surface': '#131315',
        'bg-elevated': '#1a1a1d',
        'bg-sunken': '#08080a',
        border: '#25262a',
        'border-strong': '#34353a',
        'text-primary': '#ececea',
        'text-secondary': '#b6b6b2',
        'text-muted': '#7a7a78',
        'text-faint': '#4d4d4b',
        accent: '#6c8ff5',
        'accent-strong': '#88a4f7',
        'accent-dim': '#2a3a66',
        'lr-color': '#d4a86a',
        success: '#5fa776',
        warning: '#d9a24a',
        danger: '#d96a5c'
      },
      fontFamily: {
        sans: ['Geist', 'system-ui', 'sans-serif'],
        mono: ['"Geist Mono"', 'ui-monospace', 'monospace'],
        serif: ['"Instrument Serif"', 'Iowan Old Style', 'serif']
      },
      borderRadius: {
        DEFAULT: '6px',
        sm: '3px',
        md: '6px',
        lg: '10px'
      },
      letterSpacing: {
        tightish: '-0.015em',
        tighter2: '-0.025em'
      }
    }
  },
  plugins: []
}
