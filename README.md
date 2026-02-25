<div align="center">

<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!--                          ANIMATED HEADER                              -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,25:161b22,50:6e40c9,75:58a6ff,100:79c0ff&height=260&section=header&text=🔮%20Hand-Tracked%20Glass%20Rings&fontSize=42&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Real-Time%20AR%20•%20GLSL%20Shaders%20•%20Gesture%20Recognition&descSize=18&descAlignY=55&descAlign=50" width="100%"/>

<!-- Animated typing effect -->
<a href="https://git.io/typing-svg">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=24&duration=3000&pause=1000&color=58A6FF&center=true&vCenter=true&multiline=true&repeat=true&width=700&height=100&lines=✨+Cinematic+AR+Energy+Interface;🖐️+Real-Time+Hand+Tracking+%2B+Glass+Rings;🌈+Chromatic+Aberration+%7C+Fresnel+%7C+Iridescence;⚡+Gesture-Reactive+Shader+Animations" alt="Typing SVG" />
</a>

<br/>

<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!--                         INTERACTIVE BADGES                            -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenGL](https://img.shields.io/badge/OpenGL-3.3+_Core-5586A4?style=for-the-badge&logo=opengl&logoColor=white)](https://www.opengl.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand_Tracking-0097A7?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev)
[![GLSL](https://img.shields.io/badge/GLSL-Fragment_Shaders-FF6F00?style=for-the-badge&logo=opengl&logoColor=white)](#-shader-pipeline)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](#license)

<br/>

[![Stars](https://img.shields.io/github/stars/sarthakbhopale/hand-tracked-glass-rings?style=social)](https://github.com/sarthakbhopale/hand-tracked-glass-rings)
[![Forks](https://img.shields.io/github/forks/sarthakbhopale/hand-tracked-glass-rings?style=social)](https://github.com/sarthakbhopale/hand-tracked-glass-rings)

<br/>

<!-- Animated divider -->
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

</div>

---

<div align="center">

## ✨ Overview

**A cinematic AR experience that renders real-time 3D glass torus rings on your fingers,**
**powered by GPU GLSL shaders, MediaPipe hand tracking, and gesture-reactive animations.**

<br/>

Every ring refracts light with **chromatic aberration**, shimmers with **Fresnel iridescence**,
and responds dynamically to **hand gestures** — creating a premium, cinematic visual effect.

<br/>

<!-- Feature highlights with animated icons -->
<table>
<tr>
<td align="center" width="25%">
<img src="https://img.icons8.com/fluency/96/hand.png" width="60"/>
<br/><b>Hand Tracking</b>
<br/><sub>21-point landmark detection<br/>with Kalman smoothing</sub>
</td>
<td align="center" width="25%">
<img src="https://img.icons8.com/fluency/96/crystal-ball.png" width="60"/>
<br/><b>Glass Shaders</b>
<br/><sub>6 GLSL shader modes<br/>with real-time refraction</sub>
</td>
<td align="center" width="25%">
<img src="https://img.icons8.com/fluency/96/lightning-bolt.png" width="60"/>
<br/><b>Gesture Engine</b>
<br/><sub>Static + temporal gestures<br/>with two-hand interaction</sub>
</td>
<td align="center" width="25%">
<img src="https://img.icons8.com/fluency/96/movie-projector.png" width="60"/>
<br/><b>Post-Processing</b>
<br/><sub>Bloom, chromatic aberration<br/>shockwave, vignette</sub>
</td>
</tr>
</table>

</div>

---

<div align="center">

## 🎮 Gesture Controls

<br/>

| Gesture | Effect | Visual |
|:---:|:---:|:---:|
| ✋ **Open Palm** | Rings expand outward & pulse with breathing animation | `scale 1.1→1.25` · gentle wobble |
| ✊ **Fist** | Rings contract tight & glow hot orange | `scale 0.6` · tumble spin · max glow |
| 🤏 **Pinch** | Pinched finger rings orbit rapidly like spinning coins | `360°/s X` · `540°/s Y` · `180°/s Z` |
| ✌️ **Peace** | Rainbow caustic explosion on index + middle fingers | `rainbow 1.0` · `scale 1.4` · fast wobble |
| 👆 **Point** | Directional energy beam from index finger | energy arc particles |
| 🤌 **Snap** | Shockwave ripple effect from snap point | post-processing ripple |
| 🫸🫷 **Two-Hand Pull** | Energy field stretches between hands | particle bridges |
| 🫷🫸 **Two-Hand Compress** | Energy field compresses with intensity | compressed glow |

</div>

---

<div align="center">

## 🎨 Shader Pipeline

**6 switchable GLSL fragment shader modes — all running in real-time on the GPU**

<br/>

</div>

<div align="center">

<table>
<tr>
<td align="center"><b>Mode</b></td>
<td align="center"><b>Name</b></td>
<td align="center"><b>Description</b></td>
</tr>
<tr>
<td align="center"><code>0</code></td>
<td align="center">🔮 Advanced Glass</td>
<td align="center">Multi-layer Fresnel, 5-tap chromatic aberration, FBM ripple</td>
</tr>
<tr>
<td align="center"><code>1</code></td>
<td align="center">💧 Liquid Refraction</td>
<td align="center">Flowing caustics, viscous deformation, water-like distortion</td>
</tr>
<tr>
<td align="center"><code>2</code></td>
<td align="center">⚡ Plasma Energy</td>
<td align="center">FBM emissive glow, pulsing veins, electric arc patterns</td>
</tr>
<tr>
<td align="center"><code>3</code></td>
<td align="center">📡 Holographic</td>
<td align="center">Scanlines, wireframe overlay, RGB shift, digital flicker</td>
</tr>
<tr>
<td align="center"><code>4</code></td>
<td align="center">🌀 Void Crystal</td>
<td align="center">Dark matter absorption, void distortion, crystal refraction</td>
</tr>
<tr>
<td align="center"><code>5</code></td>
<td align="center">🔥 Solar Flare</td>
<td align="center">Fire emission, solar prominance, heat haze distortion</td>
</tr>
</table>

</div>

<div align="center">

### 🔬 Core Shader Techniques

</div>

<div align="center">

```glsl
// ═══════════════════════════════════════════
//  CHROMATIC ABERRATION — Glass Dispersion
// ═══════════════════════════════════════════

float eta_R = 1.0 / 1.47;   // Red channel IOR
float eta_G = 1.0 / 1.50;   // Green channel IOR
float eta_B = 1.0 / 1.53;   // Blue channel IOR

vec3 refR = refract(-V, N, eta_R);
vec3 refG = refract(-V, N, eta_G);
vec3 refB = refract(-V, N, eta_B);

vec3 refractedColor = vec3(
    texture(bgTexture, uvR).r,   // Red from shifted UV
    texture(bgTexture, uvG).g,   // Green from shifted UV
    texture(bgTexture, uvB).b    // Blue from shifted UV
);
```

```glsl
// ═══════════════════════════════════════════
//  FRESNEL + IRIDESCENCE — Rainbow Caustics
// ═══════════════════════════════════════════

float cosTheta = max(dot(N, V), 0.0);
float fresnel = 0.04 + 0.96 * pow(1.0 - cosTheta, 5.0);

vec3 iridescence = vec3(
    0.5 + 0.5 * sin(angle * 6.2832 + 0.0),
    0.5 + 0.5 * sin(angle * 6.2832 + 2.094),
    0.5 + 0.5 * sin(angle * 6.2832 + 4.189)
);

vec3 color = mix(refractedColor, iridescence, fresnel * rainbowMix);
```

```glsl
// ═══════════════════════════════════════════
//  PROCEDURAL VERTEX DISPLACEMENT — FBM Noise
// ═══════════════════════════════════════════

float hash(float n) { return fract(sin(n) * 43758.5453123); }

float fbm(vec3 p) {
    float v = 0.0; float a = 0.5;
    for(int i = 0; i < 3; i++) {
        v += a * noise3d(p);
        p *= 2.01; a *= 0.5;
    }
    return v;
}
// Organic undulation: displace along normal
pos += norm * fbm(pos * 3.0 + uTime * 0.5) * uDeformAmount;
```

</div>

---

<div align="center">

## 🏗️ Architecture

```mermaid
graph TD
    A["📷 Webcam Feed"] --> B["🖐️ MediaPipe HandLandmarker"]
    B --> C["🎯 Gesture Engine"]
    C --> D["⚡ Energy State"]
    D --> E["💍 Ring Controller"]
    D --> F["✨ Particle System"]
    E --> G["🎨 Shader Manager"]
    G --> H["🖥️ OpenGL 3.3 Core"]
    F --> H
    H --> I["🎬 Post-Processing"]
    I --> J["📺 Final Output"]

    style A fill:#161b22,stroke:#58a6ff,color:#c9d1d9
    style B fill:#161b22,stroke:#58a6ff,color:#c9d1d9
    style C fill:#161b22,stroke:#6e40c9,color:#c9d1d9
    style D fill:#161b22,stroke:#6e40c9,color:#c9d1d9
    style E fill:#161b22,stroke:#f78166,color:#c9d1d9
    style F fill:#161b22,stroke:#f78166,color:#c9d1d9
    style G fill:#161b22,stroke:#3fb950,color:#c9d1d9
    style H fill:#161b22,stroke:#3fb950,color:#c9d1d9
    style I fill:#161b22,stroke:#d29922,color:#c9d1d9
    style J fill:#161b22,stroke:#d29922,color:#c9d1d9
```

</div>

---

<div align="center">

## 📁 Project Structure

```
hand-tracked-glass-rings/
├── 🚀 main.py               # Entry point — ties all modules together
├── 💍 glass_torus_ar.py      # Core AR renderer + torus geometry + GLSL shaders
├── 🎨 shader_manager.py      # 6 switchable GLSL shader modes
├── 🎯 gesture_engine.py      # Static + temporal gesture recognition
├── ⚡ energy_state.py        # Energy state machine for visual effects
├── 🎮 ring_controller.py     # Per-finger ring positioning & animation
├── ✨ particle_system.py     # GPU-efficient orbit particles & energy arcs
├── 🎬 post_processing.py     # FBO bloom, chromatic aberration, shockwave
├── 🧮 math_utils.py          # Matrix utilities & Kalman filters
├── 🤖 hand_landmarker.task   # MediaPipe hand tracking model
└── 🤖 face_landmarker.task   # MediaPipe face landmark model
```

</div>

---

<div align="center">

## 🚀 Quick Start

</div>

<div align="center">

### Prerequisites

**Python 3.8+** · **Webcam** · **GPU with OpenGL 3.3+ support**

<br/>

### Installation

</div>

```bash
# Clone the repository
git clone https://github.com/sarthakbhopale/hand-tracked-glass-rings.git
cd hand-tracked-glass-rings

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install opencv-python mediapipe PyOpenGL PyOpenGL-accelerate glfw numpy
```

<div align="center">

### Run

</div>

```bash
python main.py
```

<div align="center">

> **Press `ESC` to quit · Window: 1280×720 · Supports up to 2 hands simultaneously**

</div>

---

<div align="center">

## 📦 Dependencies

| Package | Purpose | Badge |
|:---:|:---:|:---:|
| OpenCV | Camera capture & image processing | ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv) |
| MediaPipe | 21-point hand landmark detection | ![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-0097A7?style=flat-square&logo=google) |
| PyOpenGL | OpenGL 3.3 Core rendering | ![OpenGL](https://img.shields.io/badge/PyOpenGL-3.1+-5586A4?style=flat-square&logo=opengl) |
| GLFW | Window management & input | ![GLFW](https://img.shields.io/badge/GLFW-3.x-FF6600?style=flat-square) |
| NumPy | Matrix math & particle simulation | ![NumPy](https://img.shields.io/badge/NumPy-1.20+-013243?style=flat-square&logo=numpy) |

</div>

---

<div align="center">

## 🔧 Technical Highlights

</div>

<div align="center">

| Feature | Details |
|:---:|:---:|
| 🎯 **Tracking** | MediaPipe Tasks API · 21 landmarks · Kalman-smoothed · 2 hands |
| 🔮 **Rendering** | OpenGL 3.3 Core Profile · VAO/VBO/EBO · Instanced torus geometry |
| 🌈 **Shaders** | 6 GLSL modes · Chromatic aberration · Fresnel · FBM noise · Vertex displacement |
| ✨ **Particles** | 800 GPU particles · Orbit dynamics · Energy arcs · Trail echoes |
| 🎬 **Post-FX** | Ping-pong FBO · Bloom · Vignette · Shockwave ripple · Film grain |
| 🎮 **Gestures** | 5 static + 4 temporal + 3 two-hand · Velocity tracking · Confidence filtering |
| ⚡ **Performance** | EMA jitter filter · Vectorized NumPy · ~30 FPS on integrated GPU |

</div>

---

<div align="center">

## 🎥 How It Works

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  📷 Camera   │ ──▶ │  🖐️ Detect    │ ──▶ │  🎯 Recognize    │
│  Capture     │     │  Hands       │     │  Gestures       │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
                    ┌──────────────┐     ┌─────────▼────────┐
                    │  🖥️ Display   │ ◀── │  🎨 Render        │
                    │  Final Frame │     │  Rings + FX      │
                    └──────────────┘     └──────────────────┘
```

**1.** Webcam captures each frame at 1280×720
**2.** MediaPipe detects hand landmarks in real-time
**3.** Gesture engine classifies poses & tracks velocity
**4.** Energy state drives shader uniforms & particle intensity
**5.** GPU renders glass torus rings with GLSL shaders
**6.** Post-processing applies bloom, chromatic aberration & vignette
**7.** Final composited frame displayed via GLFW window

</div>

---

<div align="center">

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

<br/>

<a href="https://github.com/sarthakbhopale/hand-tracked-glass-rings/issues">
  <img src="https://img.shields.io/badge/🐛_Report_Bug-d73a4a?style=for-the-badge" alt="Report Bug"/>
</a>
&nbsp;&nbsp;
<a href="https://github.com/sarthakbhopale/hand-tracked-glass-rings/issues">
  <img src="https://img.shields.io/badge/💡_Request_Feature-a2eeef?style=for-the-badge&logoColor=black" alt="Request Feature"/>
</a>
&nbsp;&nbsp;
<a href="https://github.com/sarthakbhopale/hand-tracked-glass-rings/pulls">
  <img src="https://img.shields.io/badge/🔀_Submit_PR-6f42c1?style=for-the-badge" alt="Submit PR"/>
</a>

</div>

---

<div align="center">

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

</div>

---

<div align="center">

## 👨‍💻 Author

<br/>

<img src="https://img.shields.io/badge/Made_with_❤️_by-Sarthak_Bhopale-6e40c9?style=for-the-badge&logo=github&logoColor=white" alt="Author"/>

<br/><br/>

<a href="https://github.com/sarthakbhopale">
  <img src="https://img.shields.io/badge/GitHub-sarthakbhopale-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/>
</a>

<br/><br/>

⭐ **If you found this project interesting, please consider giving it a star!** ⭐

<br/>

</div>

<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!--                          ANIMATED FOOTER                              -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:79c0ff,25:58a6ff,50:6e40c9,75:161b22,100:0d1117&height=120&section=footer" width="100%"/>

</div>
