import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        'mono': ['JetBrains Mono', 'monospace'],
      },
      colors: {
        primary: {
          bg: '#0a0a0f',
          secondary: '#12121a',
          tertiary: '#1a1a24',
        },
        text: {
          primary: '#e8e8ed',
          secondary: '#8888a0',
        },
        accent: {
          primary: '#7b68ee',
          glow: 'rgba(123, 104, 238, 0.3)',
        },
        success: '#4ade80',
        warning: '#fbbf24',
        error: '#f87171',
        border: '#2a2a3a',
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic':
          'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
      },
    },
  },
  plugins: [],
};
export default config;