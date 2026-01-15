import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
    plugins: [react()],

    // Tauri expects a static output
    clearScreen: false,

    server: {
        port: 3000,
        strictPort: true,
        // Disable HMR overlay in Tauri
        hmr: {
            overlay: false
        }
    },

    // Tauri uses a custom protocol in production
    optimizeDeps: {
        exclude: ['@react-three/postprocessing']
    },

    build: {
        target: ['es2021', 'chrome100', 'safari13'],
        minify: !process.env.TAURI_DEBUG ? 'esbuild' : false,
        sourcemap: !!process.env.TAURI_DEBUG,
    },

    // Prevent vite from obscuring rust errors
    envPrefix: ['VITE_', 'TAURI_'],
})
