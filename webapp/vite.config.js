import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Vite-Konfiguration für reine Client-App
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, open: true }
})
