/**
 * BRICK OS: Unified SDF Vertex Shader
 * 
 * Renders fullscreen quad in NDC space for raymarching.
 */

varying vec2 vUv;

void main() {
    // Manually calculate UV from position (assuming plane is -1 to 1)
    vUv = position.xy * 0.5 + 0.5;
    
    // Render directly to NDC space (fullscreen quad) at Far Plane
    // z = 0.9999 ensures it is behind everything else but NOT clipped
    gl_Position = vec4(position.xy, 0.9999, 1.0);
}
