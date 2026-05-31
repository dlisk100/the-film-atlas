/** @type {import('tailwindcss').Config} */
export default {
  content: ["./src/**/*.{astro,html,md,mdx,js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "hsl(var(--bg))",
        fg: "hsl(var(--fg))",
        muted: "hsl(var(--muted))",
        surface: "hsl(var(--surface))",
        border: "hsl(var(--border))",
        accent: "hsl(var(--accent))",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui"],
        serif: ["Source Serif 4", "Newsreader", "ui-serif", "Georgia"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular"],
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
};
