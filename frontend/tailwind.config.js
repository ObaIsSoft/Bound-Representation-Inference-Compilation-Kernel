/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {},
    },
    plugins: [],
    safelist: [
        {
            pattern: /(bg|text|border)-(emerald|blue|rose|amber|purple|slate)-(400|500|600|700|800|900|950)/,
            variants: ['hover', 'focus'],
        },
    ],
}
