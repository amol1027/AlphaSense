import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

const marketApiProxy = {
  '/market-api': {
    target: 'http://127.0.0.1:8000',
    changeOrigin: true,
    rewrite: (path: string) => path.replace(/^\/market-api/, ''),
  },
  // Same-origin proxy for the account API so the dev server never needs CORS.
  // Use VITE_AUTH_API_BASE_URL=/auth-api (see .env.example) to route through here.
  '/auth-api': {
    target: 'http://127.0.0.1:8010',
    changeOrigin: true,
    rewrite: (path: string) => path.replace(/^\/auth-api/, ''),
  },
}

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { proxy: marketApiProxy },
  preview: { proxy: marketApiProxy },
})
