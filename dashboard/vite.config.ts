import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// base './' so the same static build works on GitHub Pages (sub-path) and Vercel (root)
export default defineConfig({ plugins: [react()], base: './' });
