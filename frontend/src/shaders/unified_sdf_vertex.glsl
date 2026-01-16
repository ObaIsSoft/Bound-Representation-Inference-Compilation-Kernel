/**
 * BRICK OS: Unified SDF Vertex Shader
 * 
 * Renders fullscreen quad in NDC space for raymarching.
 */

varying vec2 vUv;

void main() {
    vUv = uv;
    // Render directly to NDC space (fullscreen quad)
    gl_Position = vec4(position, 1.0);
}
