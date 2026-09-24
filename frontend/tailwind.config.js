/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        canvas: "#F5F5F7",
        surface: "#F5F5F7",
        card: "#FFFFFF",
        apple: {
          canvas: "#F5F5F7",
          card: "#FFFFFF",
          text: "#1D1D1F",
          secondary: "#86868B",
          tertiary: "#A1A1A6",
          border: "rgba(0, 0, 0, 0.06)",
          borderStrong: "rgba(0, 0, 0, 0.12)",
          tray: "#E8E8ED",
          subtle: "#F0F0F2",
          blue: "#0071E3",
          blueHover: "#0077ED",
          blueSubtle: "#EBF5FF",
          red: "#E03E3E",
          green: "#34C759",
        },
        brand: {
          50: "#EBF5FF",
          100: "#D6EBFF",
          500: "#0071E3",
          600: "#0071E3",
          700: "#0058B0",
        },
        primary: "#1D1D1F",
        secondary: "#86868B",
        muted: "#A1A1A6",
        price: {
          up: "#E03E3E",
          down: "#34C759",
          flat: "#86868B",
        },
      },
      boxShadow: {
        apple: "0 2px 8px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.02)",
        "apple-hover": "0 8px 24px rgba(0, 0, 0, 0.06), 0 2px 6px rgba(0, 0, 0, 0.03)",
        "apple-float": "0 12px 32px rgba(0, 0, 0, 0.08), 0 2px 8px rgba(0, 0, 0, 0.04)",
      },
      borderRadius: {
        "2xl": "18px",
        "3xl": "24px",
      },
    },
  },
  plugins: [],
};
