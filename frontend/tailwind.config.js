/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            fontFamily: {
                sans: ['"League Spartan"', 'sans-serif'], // Override default sans
                majestic: ['"League Spartan"', 'sans-serif'], // Custom class
            }
        },
    },
    plugins: [],
    safelist: [
        {
            pattern: /(bg|text|border)-(emerald|blue|rose|amber|purple|slate)-(400|500|600|700|800|900|950)/,
            variants: ['hover', 'focus'],
        },
    ],
}
