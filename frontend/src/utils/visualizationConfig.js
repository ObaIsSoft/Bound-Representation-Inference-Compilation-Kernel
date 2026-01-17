/**
 * BRICK OS: Visualization Quality Configuration
 * 
 * Quality presets for physics visualizations to optimize performance
 * across different hardware capabilities.
 */

export const QUALITY_PRESETS = {
    ULTRA: {
        particleCount: 500,
        fieldLineDensity: 64,
        trailLength: 100,
        shaderPrecision: 'highp',
        antialiasing: true,
        shadowQuality: 'high',
        label: 'Ultra'
    },
    HIGH: {
        particleCount: 300,
        fieldLineDensity: 32,
        trailLength: 75,
        shaderPrecision: 'mediump',
        antialiasing: true,
        shadowQuality: 'medium',
        label: 'High'
    },
    MEDIUM: {
        particleCount: 150,
        fieldLineDensity: 16,
        trailLength: 50,
        shaderPrecision: 'mediump',
        antialiasing: false,
        shadowQuality: 'low',
        label: 'Medium'
    },
    LOW: {
        particleCount: 50,
        fieldLineDensity: 8,
        trailLength: 25,
        shaderPrecision: 'lowp',
        antialiasing: false,
        shadowQuality: 'none',
        label: 'Low'
    }
};

/**
 * Diverse Temperature Color Mapping
 * Maps temperature (Celsius) to RGB colors for thermal visualizations
 */
export const TEMPERATURE_GRADIENT = [
    { temp: -273, color: '#4A148C', label: 'Absolute Zero' },    // Deep Purple
    { temp: -100, color: '#6A1B9A', label: 'Cryogenic' },        // Purple
    { temp: -50, color: '#1976D2', label: 'Deep Freeze' },       // Deep Blue
    { temp: 0, color: '#00BCD4', label: 'Freezing' },            // Cyan
    { temp: 20, color: '#4CAF50', label: 'Room Temp' },          // Green
    { temp: 100, color: '#FFC107', label: 'Boiling' },           // Yellow
    { temp: 300, color: '#FF9800', label: 'Hot' },               // Amber
    { temp: 500, color: '#FF5722', label: 'Very Hot' },          // Deep Orange
    { temp: 1000, color: '#F44336', label: 'Glowing' },          // Red
    { temp: 1500, color: '#FFEBEE', label: 'Molten' },           // Near White
    { temp: 2000, color: '#FFFFFF', label: 'Plasma' }            // White
];

/**
 * Get color for a given temperature using linear interpolation
 * @param {number} temp - Temperature in Celsius
 * @returns {string} Hex color code
 */
export function getTempColor(temp) {
    // Clamp to range
    if (temp <= TEMPERATURE_GRADIENT[0].temp) {
        return TEMPERATURE_GRADIENT[0].color;
    }
    if (temp >= TEMPERATURE_GRADIENT[TEMPERATURE_GRADIENT.length - 1].temp) {
        return TEMPERATURE_GRADIENT[TEMPERATURE_GRADIENT.length - 1].color;
    }

    // Find surrounding gradient stops
    for (let i = 0; i < TEMPERATURE_GRADIENT.length - 1; i++) {
        const stop1 = TEMPERATURE_GRADIENT[i];
        const stop2 = TEMPERATURE_GRADIENT[i + 1];

        if (temp >= stop1.temp && temp <= stop2.temp) {
            // Linear interpolation
            const ratio = (temp - stop1.temp) / (stop2.temp - stop1.temp);
            return interpolateColor(stop1.color, stop2.color, ratio);
        }
    }

    return TEMPERATURE_GRADIENT[0].color; // Fallback
}

/**
 * Interpolate between two hex colors
 */
function interpolateColor(color1, color2, ratio) {
    const hex1 = color1.replace('#', '');
    const hex2 = color2.replace('#', '');

    const r1 = parseInt(hex1.substring(0, 2), 16);
    const g1 = parseInt(hex1.substring(2, 4), 16);
    const b1 = parseInt(hex1.substring(4, 6), 16);

    const r2 = parseInt(hex2.substring(0, 2), 16);
    const g2 = parseInt(hex2.substring(2, 4), 16);
    const b2 = parseInt(hex2.substring(4, 6), 16);

    const r = Math.round(r1 + (r2 - r1) * ratio);
    const g = Math.round(g1 + (g2 - g1) * ratio);
    const b = Math.round(b1 + (b2 - b1) * ratio);

    return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
}
