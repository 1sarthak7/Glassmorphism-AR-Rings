"""
Shader Manager
==============
6 GLSL shader modes with advanced effects: procedural noise, animated distortion,
multi-layer Fresnel, chromatic aberration, vertex displacement.

Modes: Advanced Glass, Liquid Refraction, Plasma Energy,
       Holographic, Dark Matter, Space-Time Distortion.
"""

from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

# ============================================================================
#  COMMON VERTEX SHADER (shared by all ring modes)
# ============================================================================
#
# Features:
#   - Standard MVP transform
#   - Procedural vertex displacement via noise (organic undulation)
#   - Elastic bending toward a target point (hand position)
#   - Outputs: world position, normal, clip position, local UV

COMMON_VERTEX = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float uTime;
uniform float uDeformAmount;    // 0..1 vertex noise deformation
uniform float uEnergy;          // 0..1 energy level
uniform vec3 uBendTarget;       // world-space point to bend toward

out vec3 FragPos;
out vec3 Normal;
out vec4 ClipPos;
out vec2 LocalUV;               // parametric UV on torus surface

// --- Simple hash-based noise for vertex displacement ---
float hash(float n) { return fract(sin(n) * 43758.5453123); }
float noise3d(vec3 p) {
    vec3 ip = floor(p);
    vec3 fp = fract(p);
    fp = fp * fp * (3.0 - 2.0 * fp);
    float n = ip.x + ip.y * 57.0 + ip.z * 113.0;
    return mix(mix(mix(hash(n), hash(n+1.0), fp.x),
               mix(hash(n+57.0), hash(n+58.0), fp.x), fp.y),
           mix(mix(hash(n+113.0), hash(n+114.0), fp.x),
               mix(hash(n+170.0), hash(n+171.0), fp.x), fp.y), fp.z);
}

float fbm(vec3 p) {
    float v = 0.0; float a = 0.5;
    for(int i = 0; i < 3; i++) {
        v += a * noise3d(p);
        p *= 2.01; a *= 0.5;
    }
    return v;
}

void main() {
    vec3 pos = aPos;
    vec3 norm = aNormal;

    // --- Procedural vertex displacement ---
    if (uDeformAmount > 0.001) {
        float n = fbm(pos * 3.0 + uTime * 0.5) * 2.0 - 1.0;
        // Organic undulation: displace along normal
        pos += norm * n * uDeformAmount * 0.15;
        // Energy-driven pulse
        float pulse = sin(uTime * 4.0 + pos.y * 6.0) * uEnergy * uDeformAmount * 0.05;
        pos += norm * pulse;
    }

    vec4 worldPos = model * vec4(pos, 1.0);

    // --- Elastic bending toward target ---
    // Gently pull vertices toward hand center
    if (uDeformAmount > 0.001) {
        vec3 toTarget = uBendTarget - worldPos.xyz;
        float dist = length(toTarget);
        if (dist > 0.01) {
            float bendStrength = uDeformAmount * 0.1 / (1.0 + dist);
            worldPos.xyz += toTarget * bendStrength;
        }
    }

    FragPos = worldPos.xyz;
    Normal = normalize(mat3(transpose(inverse(model))) * norm);
    ClipPos = projection * view * worldPos;
    gl_Position = ClipPos;

    // Parametric UV from local position (approximate)
    LocalUV = pos.xy * 0.5 + 0.5;
}
"""

# ============================================================================
#  BACKGROUND SHADERS (passthrough)
# ============================================================================

BG_VERTEX = """
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;
out vec2 TexCoord;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoord = aTexCoord;
}
"""

BG_FRAGMENT = """
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D bgTexture;
void main() {
    FragColor = texture(bgTexture, TexCoord);
}
"""

# ============================================================================
#  FRAGMENT SHADERS — 6 MODES
# ============================================================================

# --- COMMON GLSL FUNCTIONS (prepended to all fragment shaders) ---
FRAG_COMMON = """
#version 330 core

in vec3 FragPos;
in vec3 Normal;
in vec4 ClipPos;
in vec2 LocalUV;

out vec4 FragColor;

uniform sampler2D bgTexture;
uniform vec3 cameraPos;
uniform vec3 lightDir;
uniform float uTime;
uniform float uEnergy;
uniform float uGlow;
uniform float uRainbow;
uniform float uPulse;

// --- Shared utility functions ---
float hash11(float p) { return fract(sin(p) * 43758.5453123); }
float hash21(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

float noise3d(vec3 p) {
    vec3 ip = floor(p); vec3 fp = fract(p);
    fp = fp*fp*(3.0-2.0*fp);
    float n = ip.x + ip.y*57.0 + ip.z*113.0;
    return mix(mix(mix(hash11(n), hash11(n+1.0), fp.x),
               mix(hash11(n+57.0), hash11(n+58.0), fp.x), fp.y),
           mix(mix(hash11(n+113.0), hash11(n+114.0), fp.x),
               mix(hash11(n+170.0), hash11(n+171.0), fp.x), fp.y), fp.z);
}

float fbm(vec3 p, int oct) {
    float v = 0.0, a = 0.5;
    for(int i = 0; i < oct; i++) { v += a * noise3d(p); p *= 2.01; a *= 0.5; }
    return v;
}

vec2 screenUV() { return (ClipPos.xy / ClipPos.w) * 0.5 + 0.5; }

float fresnelSchlick(vec3 N, vec3 V, float F0) {
    return F0 + (1.0 - F0) * pow(1.0 - max(dot(N, V), 0.0), 5.0);
}

vec3 chromaticSample(vec2 uv, vec3 N, vec3 V, float dist) {
    // IOR spread controlled by uRainbow:
    //   uRainbow=0 → all channels same IOR (no color split)
    //   uRainbow=1 → full prismatic spread (1.45 to 1.55)
    float spread = uRainbow * 0.05;
    float baseIOR = 1.50;
    vec3 rR = refract(-V, N, 1.0/(baseIOR - spread));
    vec3 rG = refract(-V, N, 1.0/baseIOR);
    vec3 rB = refract(-V, N, 1.0/(baseIOR + spread));
    return vec3(
        texture(bgTexture, clamp(uv + rR.xy*dist, 0.0, 1.0)).r,
        texture(bgTexture, clamp(uv + rG.xy*dist, 0.0, 1.0)).g,
        texture(bgTexture, clamp(uv + rB.xy*dist, 0.0, 1.0)).b
    );
}

vec3 iridescence(float angle) {
    return vec3(
        0.5 + 0.5 * sin(angle * 6.2832),
        0.5 + 0.5 * sin(angle * 6.2832 + 2.094),
        0.5 + 0.5 * sin(angle * 6.2832 + 4.189)
    );
}
"""

# ------------------------------------------------------------------
# MODE 0: ADVANCED GLASS — Multi-layer Fresnel, 5-tap chromatic, FBM ripple
# ------------------------------------------------------------------
FRAG_GLASS = FRAG_COMMON + """
void main() {
    vec3 N = normalize(Normal);
    vec3 V = normalize(cameraPos - FragPos);
    vec3 L = normalize(-lightDir);
    vec2 uv = screenUV();

    // Animated surface ripple
    float ripple = fbm(FragPos * 5.0 + uTime * 0.8, 4) * 0.015;
    vec2 rippleUV = uv + N.xy * ripple;

    // --- Refraction ---
    // When uRainbow = 0: single clean refraction (no color split)
    // When uRainbow > 0: chromatic aberration ramps up
    float refDist = 0.08 + uEnergy * 0.04;
    vec3 refracted;
    if (uRainbow < 0.05) {
        // Clean glass — single IOR, no rainbow
        vec3 R = refract(-V, N, 1.0/1.50);
        vec2 refUV = clamp(rippleUV + R.xy * refDist, 0.0, 1.0);
        refracted = texture(bgTexture, refUV).rgb;
    } else {
        // Rainbow mode — progressive chromatic split
        refracted = chromaticSample(rippleUV, N, V, refDist);
        vec3 refracted2 = chromaticSample(rippleUV + vec2(0.003), N, V, refDist * 1.1);
        refracted = mix(refracted, refracted2, uRainbow * 0.4);
    }

    float fresnel = fresnelSchlick(N, V, 0.04);

    // --- Tint: neutral by default, iridescent with rainbow ---
    vec3 glassTint = vec3(0.95, 0.97, 1.0);  // Very subtle cool tint
    if (uRainbow > 0.05) {
        vec3 rainbow = iridescence(dot(N, V) + uTime * 0.2);
        glassTint = mix(glassTint, rainbow, uRainbow * 0.6);
    }
    vec3 color = mix(refracted, glassTint, fresnel * 0.35);

    // --- Specular highlight (always white, clean) ---
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 256.0);
    color += vec3(1.0) * spec * (0.7 + uGlow * 0.5);

    // --- Rim glow: subtle ice-blue default, colorful with rainbow ---
    float rim = pow(1.0 - max(dot(N, V), 0.0), 3.0);
    vec3 rimColor = vec3(0.7, 0.8, 1.0);  // Ice blue default
    if (uRainbow > 0.05) {
        vec3 rainbowRim = iridescence(dot(N, V) * 2.0 + uTime * 0.5);
        rimColor = mix(rimColor, rainbowRim, uRainbow);
    }
    if (uGlow > 0.1) {
        rimColor = mix(rimColor, vec3(1.0, 0.6, 0.2), uGlow);
    }
    color += rimColor * rim * (0.2 + uGlow * 0.6 + uRainbow * 0.4);

    float alpha = 0.06 + fresnel * 0.4 + rim * 0.15;
    alpha *= (0.9 + 0.1 * sin(uPulse));
    FragColor = vec4(color, alpha);
}
"""

# ------------------------------------------------------------------
# MODE 1: LIQUID REFRACTION — flowing caustics, viscous deformation
# ------------------------------------------------------------------
FRAG_LIQUID = FRAG_COMMON + """
void main() {
    vec3 N = normalize(Normal);
    vec3 V = normalize(cameraPos - FragPos);
    vec3 L = normalize(-lightDir);
    vec2 uv = screenUV();

    // Flowing caustic distortion
    float flow = fbm(FragPos * 4.0 + vec3(uTime * 0.3, uTime * 0.5, uTime * 0.2), 5);
    float caustic = pow(abs(sin(flow * 12.0 + uTime * 2.0)), 3.0);

    // Viscous refraction (stronger IOR)
    float eta = 1.0 / (1.33 + 0.1 * sin(uTime + FragPos.x * 3.0));
    vec3 R = refract(-V, N, eta);
    vec2 distortedUV = clamp(uv + R.xy * 0.15 + flow * 0.03, 0.0, 1.0);
    vec3 refracted = texture(bgTexture, distortedUV).rgb;

    // Water tint
    vec3 waterTint = vec3(0.3, 0.6, 0.9);
    float fresnel = fresnelSchlick(N, V, 0.02);
    vec3 color = mix(refracted, waterTint, fresnel * 0.3 + 0.05);

    // Caustic highlights
    color += vec3(0.8, 0.9, 1.0) * caustic * 0.4 * (1.0 + uEnergy);

    // Specular
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 128.0);
    color += vec3(1.0) * spec * 0.6;

    // Surface gloss
    float rim = pow(1.0 - max(dot(N, V), 0.0), 2.0);
    color += vec3(0.5, 0.7, 1.0) * rim * 0.2;

    float alpha = 0.12 + fresnel * 0.4 + caustic * 0.1;
    FragColor = vec4(color, alpha);
}
"""

# ------------------------------------------------------------------
# MODE 2: PLASMA ENERGY — FBM emissive, pulsing veins, electric arcs
# ------------------------------------------------------------------
FRAG_PLASMA = FRAG_COMMON + """
void main() {
    vec3 N = normalize(Normal);
    vec3 V = normalize(cameraPos - FragPos);
    vec2 uv = screenUV();

    // Plasma FBM pattern
    float plasma1 = fbm(FragPos * 3.0 + uTime * vec3(0.5, 0.3, 0.7), 5);
    float plasma2 = fbm(FragPos * 6.0 - uTime * vec3(0.3, 0.6, 0.4), 4);
    float plasma = plasma1 * 0.6 + plasma2 * 0.4;

    // Energy veins — sharp ridges
    float veins = pow(abs(sin(plasma * 12.0 + uTime * 3.0)), 8.0);

    // Electric arc pattern
    float arc = pow(abs(sin(FragPos.x * 20.0 + uTime * 5.0 + plasma * 10.0)), 20.0);

    // Emissive color (hot plasma gradient)
    vec3 coldColor = vec3(0.1, 0.2, 0.8);   // Deep blue
    vec3 hotColor = vec3(1.0, 0.3, 0.05);    // Orange
    vec3 coreColor = vec3(1.0, 0.95, 0.8);   // White-hot
    float heat = plasma * 0.5 + uEnergy * 0.5;
    vec3 emissive = mix(coldColor, hotColor, heat);
    emissive = mix(emissive, coreColor, veins * (0.5 + uGlow));

    // Faint refraction underneath
    vec3 refracted = chromaticSample(uv, N, V, 0.06);
    vec3 color = mix(refracted * 0.3, emissive, 0.6 + uEnergy * 0.3);

    // Arc highlights
    color += vec3(0.8, 0.7, 1.0) * arc * 0.5;

    float fresnel = fresnelSchlick(N, V, 0.1);
    float rim = pow(1.0 - max(dot(N, V), 0.0), 2.5);
    color += emissive * rim * 0.5;

    float alpha = 0.3 + fresnel * 0.3 + veins * 0.2 + uEnergy * 0.2;
    FragColor = vec4(color, min(alpha, 0.95));
}
"""

# ------------------------------------------------------------------
# MODE 3: HOLOGRAPHIC — scanlines, wireframe, RGB shift, flicker
# ------------------------------------------------------------------
FRAG_HOLOGRAPHIC = FRAG_COMMON + """
void main() {
    vec3 N = normalize(Normal);
    vec3 V = normalize(cameraPos - FragPos);
    vec2 uv = screenUV();

    // Scanline effect (horizontal lines)
    float scanline = step(0.5, fract(gl_FragCoord.y * 0.5 + uTime * 2.0));
    float thinLine = step(0.95, fract(gl_FragCoord.y * 2.0));

    // Holographic flicker
    float flicker = 0.85 + 0.15 * hash11(floor(uTime * 15.0));

    // RGB shift (separated channels)
    float shift = 0.008 + 0.004 * sin(uTime * 3.0);
    float r = texture(bgTexture, uv + vec2(shift, 0.0)).r;
    float g = texture(bgTexture, uv).g;
    float b = texture(bgTexture, uv - vec2(shift, 0.0)).b;
    vec3 refracted = vec3(r, g, b);

    // Holographic base color (cyan-blue)
    vec3 holoColor = vec3(0.2, 0.8, 1.0);
    float fresnel = fresnelSchlick(N, V, 0.1);

    vec3 color = mix(refracted * 0.4, holoColor, fresnel * 0.5 + 0.2);

    // Wireframe-like edge highlight
    float edge = pow(1.0 - abs(dot(N, V)), 4.0);
    color += vec3(0.3, 0.9, 1.0) * edge * 0.8;

    // Scanline overlay
    color *= (0.8 + 0.2 * scanline);
    color += vec3(0.4, 0.8, 1.0) * thinLine * 0.3;

    // Data noise overlay
    float dataNoise = hash21(gl_FragCoord.xy + uTime * 100.0);
    color += vec3(0.0, 0.5, 0.8) * dataNoise * 0.05;

    color *= flicker;

    float alpha = 0.15 + fresnel * 0.35 + edge * 0.2;
    alpha *= (0.9 + 0.1 * scanline);
    FragColor = vec4(color, alpha);
}
"""

# ------------------------------------------------------------------
# MODE 4: DARK MATTER — inverted refraction, gravitational lensing, void
# ------------------------------------------------------------------
FRAG_DARK_MATTER = FRAG_COMMON + """
void main() {
    vec3 N = normalize(Normal);
    vec3 V = normalize(cameraPos - FragPos);
    vec3 L = normalize(-lightDir);
    vec2 uv = screenUV();

    // Inverted refraction: pull light inward (negative offset)
    vec3 R = refract(-V, N, 1.0 / 0.7);  // Strong inverse IOR
    vec2 lensedUV = clamp(uv - R.xy * 0.18, 0.0, 1.0);

    // Gravitational lensing: swirl distortion
    vec2 center = vec2(0.5);
    vec2 delta = uv - center;
    float dist = length(delta);
    float angle = atan(delta.y, delta.x);
    float swirl = 0.15 / (dist + 0.1) * uEnergy;
    lensedUV += vec2(cos(angle + swirl), sin(angle + swirl)) * 0.02;
    lensedUV = clamp(lensedUV, 0.0, 1.0);

    vec3 refracted = texture(bgTexture, lensedUV).rgb;

    // Darken absorbed light
    float absorption = 1.0 - pow(max(dot(N, V), 0.0), 0.5);
    refracted *= (1.0 - absorption * 0.7);

    // Dark matter glow (deep purple)
    float fresnel = fresnelSchlick(N, V, 0.08);
    vec3 voidColor = vec3(0.15, 0.0, 0.3);
    vec3 edgeColor = vec3(0.5, 0.1, 0.8);
    vec3 color = mix(refracted, voidColor, absorption * 0.4);

    float rim = pow(1.0 - max(dot(N, V), 0.0), 3.0);
    color += edgeColor * rim * (0.5 + uGlow * 0.5);

    // Event horizon glow
    float noise = fbm(FragPos * 4.0 + uTime * 0.3, 3);
    color += vec3(0.3, 0.0, 0.5) * noise * rim * 0.4;

    // Specular (dim)
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 64.0);
    color += vec3(0.4, 0.2, 0.8) * spec * 0.3;

    float alpha = 0.2 + fresnel * 0.4 + absorption * 0.2;
    FragColor = vec4(color, alpha);
}
"""

# ------------------------------------------------------------------
# MODE 5: SPACE-TIME DISTORTION — gravitational waves, time spiral, tear
# ------------------------------------------------------------------
FRAG_SPACETIME = FRAG_COMMON + """
void main() {
    vec3 N = normalize(Normal);
    vec3 V = normalize(cameraPos - FragPos);
    vec2 uv = screenUV();

    // Gravitational wave distortion (concentric ripples from center)
    vec2 center = uv - 0.5;
    float dist = length(center);
    float wave = sin(dist * 30.0 - uTime * 4.0) * 0.02 / (dist + 0.3);
    wave *= uEnergy;
    vec2 warpedUV = uv + normalize(center + 0.001) * wave;

    // Time spiral UV warp
    float spiralAngle = dist * 10.0 - uTime * 2.0;
    vec2 spiral = vec2(cos(spiralAngle), sin(spiralAngle)) * 0.01 * uEnergy;
    warpedUV += spiral;
    warpedUV = clamp(warpedUV, 0.0, 1.0);

    // Chromatic split through warped UVs
    float shift = 0.006 + 0.004 * uEnergy;
    vec3 refracted = vec3(
        texture(bgTexture, warpedUV + vec2(shift, 0.0)).r,
        texture(bgTexture, warpedUV).g,
        texture(bgTexture, warpedUV - vec2(shift, 0.0)).b
    );

    // Space-time color tint
    float fresnel = fresnelSchlick(N, V, 0.06);
    vec3 stColor = vec3(0.3, 0.5, 1.0);   // Blue shift
    vec3 color = mix(refracted, stColor, fresnel * 0.3);

    // Dimensional tear at edges
    float rim = pow(1.0 - max(dot(N, V), 0.0), 4.0);
    float tear = fbm(FragPos * 8.0 + uTime * 0.5, 4);
    vec3 tearColor = mix(vec3(0.2, 0.4, 1.0), vec3(1.0, 0.8, 0.3), tear);
    color += tearColor * rim * (0.4 + uGlow * 0.5);

    // Grid overlay (spacetime fabric visualization)
    float gridX = step(0.97, fract(FragPos.x * 10.0 + uTime * 0.1));
    float gridY = step(0.97, fract(FragPos.y * 10.0 + uTime * 0.1));
    color += vec3(0.3, 0.5, 0.8) * (gridX + gridY) * 0.15;

    // Time dilation glow
    float timePulse = sin(uTime * 3.0 + dist * 20.0) * 0.5 + 0.5;
    color += stColor * timePulse * rim * 0.2;

    float alpha = 0.1 + fresnel * 0.4 + rim * 0.25;
    FragColor = vec4(color, alpha);
}
"""

# ============================================================================
#  SHADER MODE LIST
# ============================================================================

FRAGMENT_SHADERS = [
    FRAG_GLASS,
    FRAG_LIQUID,
    FRAG_PLASMA,
    FRAG_HOLOGRAPHIC,
    FRAG_DARK_MATTER,
    FRAG_SPACETIME,
]

SHADER_NAMES = [
    "Advanced Glass",
    "Liquid Refraction",
    "Plasma Energy",
    "Holographic",
    "Dark Matter",
    "Space-Time Distortion",
]


# ============================================================================
#  SHADER MANAGER CLASS
# ============================================================================

class ShaderManager:
    """
    Compiles and manages all shader programs.
    Provides uniform setting helpers and mode switching.
    """

    def __init__(self):
        self.bg_shader = None
        self.ring_shaders = []   # 6 compiled programs
        self._current_mode = 0

    def compile_all(self):
        """Compile background + all 6 ring shader programs."""
        self.bg_shader = compileProgram(
            compileShader(BG_VERTEX, GL_VERTEX_SHADER),
            compileShader(BG_FRAGMENT, GL_FRAGMENT_SHADER),
            validate=False,
        )

        for i, frag_src in enumerate(FRAGMENT_SHADERS):
            try:
                vert = compileShader(COMMON_VERTEX, GL_VERTEX_SHADER)
                frag = compileShader(frag_src, GL_FRAGMENT_SHADER)
                prog = compileProgram(vert, frag, validate=False)
                self.ring_shaders.append(prog)
                print(f"  Compiled shader [{i}]: {SHADER_NAMES[i]}")
            except Exception as e:
                print(f"  ERROR compiling shader [{i}] {SHADER_NAMES[i]}: {e}")
                # Fallback to glass shader
                if i > 0 and len(self.ring_shaders) > 0:
                    self.ring_shaders.append(self.ring_shaders[0])
                else:
                    raise

    @property
    def current_mode(self):
        return self._current_mode

    @current_mode.setter
    def current_mode(self, idx):
        self._current_mode = idx % len(self.ring_shaders)

    @property
    def current_ring_shader(self):
        return self.ring_shaders[self._current_mode]

    @property
    def current_mode_name(self):
        return SHADER_NAMES[self._current_mode]

    def get_uniform(self, name):
        """Get uniform location from current ring shader."""
        return glGetUniformLocation(self.current_ring_shader, name)

    def use_ring_shader(self):
        glUseProgram(self.current_ring_shader)

    def use_bg_shader(self):
        glUseProgram(self.bg_shader)
