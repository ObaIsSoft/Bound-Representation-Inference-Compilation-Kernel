
uniform int uViewMode; // 0=Realistic, 1=Wire, 2=Xray, 4=Thermal, 7=Stress
uniform vec3 uBaseColor;
uniform float uMetalness;
uniform float uRoughness;

// Physics Data
uniform float uThermalTemp;
uniform float uStressLevel;
uniform float uTime;

varying vec2 vUv;
varying vec3 vNormal;
varying vec3 vWorldPosition;
varying vec3 vViewPosition;

// Constants
const vec3 LIGHT_DIR = normalize(vec3(1.0, 2.0, 3.0));

// Thermal Gradient Helper
vec3 getThermalColor(float temp) {
    if (temp < 30.0) return mix(vec3(0.0, 0.0, 0.8), vec3(0.0, 0.5, 1.0), temp / 30.0);
    if (temp < 60.0) return mix(vec3(0.0, 0.5, 1.0), vec3(0.0, 1.0, 0.5), (temp - 30.0) / 30.0);
    if (temp < 100.0) return mix(vec3(0.0, 1.0, 0.5), vec3(1.0, 1.0, 0.0), (temp - 60.0) / 40.0);
    if (temp < 200.0) return mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.5, 0.0), (temp - 100.0) / 100.0);
    return mix(vec3(1.0, 0.5, 0.0), vec3(1.0, 0.0, 0.0), clamp((temp - 200.0) / 300.0, 0.0, 1.0));
}

// Stress Gradient Helper (Cantilever approx based on X)
vec3 getStressColor(float stress, float xPos) {
    // Gradient blue (low) -> red (high)
    vec3 cBlue = vec3(0.0, 0.2, 1.0);
    vec3 cGreen = vec3(0.0, 1.0, 0.2);
    vec3 cRed = vec3(1.0, 0.0, 0.0);
    
    // Fake local variation
    float localStress = stress * (1.0 - clamp((xPos + 50.0) / 100.0, 0.0, 1.0)); 
    
    if (localStress < 0.5) return mix(cBlue, cGreen, localStress * 2.0);
    return mix(cGreen, cRed, (localStress - 0.5) * 2.0);
}

void main() {
    vec3 normal = normalize(vNormal);
    vec3 viewDir = normalize(vViewPosition); // Approximation
    
    vec3 color = uBaseColor;
    float alpha = 1.0;
    
    // ------------------------------------------------
    // MODE SWITCH
    // ------------------------------------------------
    
    if (uViewMode == 2) { // X-Ray
        float rim = 1.0 - abs(dot(normal, viewDir));
        float opacity = pow(rim, 3.0);
        color = mix(vec3(0.0, 0.1, 0.2), vec3(0.4, 0.7, 1.0), opacity);
        alpha = 0.3 + 0.7 * opacity;
    } 
    else if (uViewMode == 4) { // Thermal
        // Use uniform temp from backend
        color = getThermalColor(uThermalTemp);
        
        // Add some surface shading
        float diff = max(dot(normal, LIGHT_DIR), 0.0);
        color *= (0.5 + 0.5 * diff);
    }
    else if (uViewMode == 7) { // Stress
        // Use uniform stress from backend
        color = getStressColor(uStressLevel, vWorldPosition.x);
        
        // Warning flash
        if (uStressLevel > 0.95) {
             float flash = sin(uTime * 10.0) * 0.5 + 0.5;
             color = mix(color, vec3(1.0), flash * 0.5);
        }
    }
    else { // Realistic / Default
        float diff = max(dot(normal, LIGHT_DIR), 0.0);
        color = uBaseColor * (0.2 + 0.8 * diff);
    }

    gl_FragColor = vec4(color, alpha);
}
