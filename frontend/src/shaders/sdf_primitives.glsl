/**
 * BRICK OS: SDF Primitives Library
 * 
 * Reusable signed distance functions for geometric primitives.
 * Can be #included in main shaders.
 */

// ============================================
// BASIC PRIMITIVES
// ============================================

float sdBox(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float sdSphere(vec3 p, float r) {
    return length(p) - r;
}

float sdCylinder(vec3 p, float h, float r) {
    vec2 d = abs(vec2(length(p.xz), p.y)) - vec2(r, h);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

float sdTorus(vec3 p, vec2 t) {
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

float sdCone(vec3 p, vec2 c, float h) {
    float q = length(p.xz);
    return max(dot(c.xy, vec2(q, p.y)), -h - p.y);
}

float sdPlane(vec3 p, vec3 n, float d) {
    return dot(p, n) + d;
}

// ============================================
// BOOLEAN OPERATIONS
// ============================================

float opUnion(float d1, float d2) {
    return min(d1, d2);
}

float opSubtract(float d1, float d2) {
    return max(d1, -d2);
}

float opIntersect(float d1, float d2) {
    return max(d1, d2);
}

// ============================================
// SMOOTH BOOLEAN OPERATIONS
// ============================================

float opSmoothUnion(float d1, float d2, float k) {
    float h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) - k * h * (1.0 - h);
}

float opSmoothSubtract(float d1, float d2, float k) {
    float h = clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0);
    return mix(d1, -d2, h) + k * h * (1.0 - h);
}

float opSmoothIntersect(float d1, float d2, float k) {
    float h = clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) + k * h * (1.0 - h);
}

// ============================================
// TRANSFORMATIONS
// ============================================

vec3 opTranslate(vec3 p, vec3 offset) {
    return p - offset;
}

vec3 opRotateX(vec3 p, float angle) {
    float c = cos(angle), s = sin(angle);
    return vec3(p.x, c * p.y - s * p.z, s * p.y + c * p.z);
}

vec3 opRotateY(vec3 p, float angle) {
    float c = cos(angle), s = sin(angle);
    return vec3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
}

vec3 opRotateZ(vec3 p, float angle) {
    float c = cos(angle), s = sin(angle);
    return vec3(c * p.x - s * p.y, s * p.x + c * p.y, p.z);
}

float opRound(float d, float r) {
    return d - r;
}

float opOnion(float d, float thickness) {
    return abs(d) - thickness;
}

// ============================================
// LATTICE STRUCTURES (Manufacturing)
// ============================================

float sdGyroid(vec3 p, float scale, float thickness) {
    p *= scale;
    float g = sin(p.x) * cos(p.y) + sin(p.y) * cos(p.z) + sin(p.z) * cos(p.x);
    return abs(g) / scale - thickness;
}

float sdSchwarzP(vec3 p, float scale, float thickness) {
    p *= scale;
    float s = cos(p.x) + cos(p.y) + cos(p.z);
    return abs(s) / scale - thickness;
}

float sdDiamond(vec3 p, float scale, float thickness) {
    p *= scale;
    float d = sin(p.x) * sin(p.y) * sin(p.z) + 
              sin(p.x) * cos(p.y) * cos(p.z) +
              cos(p.x) * sin(p.y) * cos(p.z) +
              cos(p.x) * cos(p.y) * sin(p.z);
    return abs(d) / scale - thickness;
}

// ============================================
// TOOLPATH (for VMK machining)
// ============================================

float sdToolPath(vec3 p, vec3 start, vec3 end, float radius) {
    return sdCapsule(p, start, end, radius);
}

// Multiple toolpaths (subtractive manufacturing)
float opToolPaths(float baseSDF, vec3 p, vec3 starts[50], vec3 ends[50], float radii[50], int count) {
    float cuts = 1e10;
    for (int i = 0; i < 50; i++) {
        if (i >= count) break;
        float cut = sdToolPath(p, starts[i], ends[i], radii[i]);
        cuts = min(cuts, cut);
    }
    return max(baseSDF, -cuts); // Boolean subtract
}
