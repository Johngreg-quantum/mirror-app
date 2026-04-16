# Mirror App — Design System Specification

## Creative North Star: "The Cinematic Astral"

This design system transports the user from a standard interface into an **immersive, high-fidelity cinematic experience**. The Creative North Star is **"The Cinematic Astral."**

Unlike traditional applications that rely on flat grids and rigid layouts, this system uses the depth of the cosmos to create a sense of boundless potential. The UI is treated as a series of **floating, translucent HUD (Heads-Up Display) elements** suspended in 3D space. Through intentional asymmetry — such as overlapping thumbnails and dynamic glowing waveforms — the design evokes the feeling of a premium film editing suite or a futuristic space vessel's interface.

---

## 1. Color Palette

The palette is built on a foundational void of deep space, punctuated by high-energy neon pulses.

### Foundation Colors

| Token                       | Hex         | Usage                              |
|-----------------------------|-------------|-------------------------------------|
| `--surface`                 | `#0b0e14`   | Primary background, canvas          |
| `--surface-dim`             | `#0b0e14`   | Dimmed surfaces                     |
| `--surface-bright`          | `#282c36`   | Elevated/bright surface areas       |
| `--surface-container-lowest`| `#000000`   | Deepest recessed areas              |
| `--surface-container-low`   | `#10131a`   | Low-elevation containers            |
| `--surface-container`       | `#161a21`   | Default container background        |
| `--surface-container-high`  | `#1c2028`   | High-elevation containers           |
| `--surface-container-highest`| `#22262f`  | Highest elevation containers        |
| `--surface-variant`         | `#22262f`   | Alternative surface, glassmorphism  |

### Accent Tiers (Neon)

| Tier          | Token          | Hex         | Symbolism                             |
|---------------|----------------|-------------|---------------------------------------|
| **Primary**   | `--primary`    | `#81ecff`   | Cyan — cool precision of advanced tech|
| **Primary Dim**| `--primary-dim`| `#00d4ec`  | Muted primary state                   |
| **Primary Container** | `--primary-container` | `#00e3fd` | Active states, CTAs gradient |
| **Secondary** | `--secondary`  | `#ffd709`   | Gold — warmth of a rising sun         |
| **Secondary Dim** | `--secondary-dim` | `#efc900` | Muted gold                        |
| **Secondary Container** | `--secondary-container` | `#705d00` | Gold container   |
| **Tertiary**  | `--tertiary`   | `#ff7350`   | Red/Orange — high-energy mastery      |
| **Tertiary Dim** | `--tertiary-dim` | `#dc3300` | Muted red                          |
| **Tertiary Container** | `--tertiary-container` | `#fc3c00` | Red container    |

### Text / On-Surface Colors

| Token                  | Hex         | Usage                        |
|------------------------|-------------|-------------------------------|
| `--on-background`      | `#ecedf6`   | Primary text on backgrounds   |
| `--on-surface`         | `#ecedf6`   | Primary text on surfaces      |
| `--on-surface-variant`  | `#a9abb3`   | Subdued/secondary text        |
| `--on-primary`         | `#005762`   | Text on primary accent areas  |
| `--on-secondary`       | `#5b4b00`   | Text on secondary accent areas|
| `--on-tertiary`        | `#440900`   | Text on tertiary accent areas |

### Error / Feedback

| Token              | Hex         |
|--------------------|-------------|
| `--error`          | `#ff716c`   |
| `--error-dim`      | `#d7383b`   |
| `--error-container`| `#9f0519`   |
| `--on-error`       | `#490006`   |

### Structural

| Token              | Hex         | Usage                              |
|--------------------|-------------|-------------------------------------|
| `--outline`        | `#73757d`   | Standard outlines                   |
| `--outline-variant`| `#45484f`   | Ghost borders (15% opacity)         |
| `--inverse-surface`| `#f9f9ff`   | Inverted surface contexts           |
| `--inverse-primary`| `#006976`   | Inverted primary                    |

---

## 2. Typography: Editorial Futurism

A high-contrast pairing that balances technical precision with modern readability.

### Font Stack

| Role              | Font Family           | Weight(s)        | Usage                          |
|-------------------|-----------------------|------------------|---------------------------------|
| **Display / Headlines** | Space Grotesk    | 700, 600         | Hero moments, h1-h3, titles   |
| **Body / Labels**       | Plus Jakarta Sans| 400, 500, 600    | Paragraphs, labels, metadata  |

### Type Scale

| Token          | Size     | Line Height | Letter Spacing | Usage                       |
|----------------|----------|-------------|----------------|-----------------------------|
| `display-lg`   | `3.5rem` | `1.1`       | `-0.02em`      | Hero / splash text          |
| `display-md`   | `2.75rem`| `1.15`      | `-0.02em`      | Section heroes              |
| `display-sm`   | `2.25rem`| `1.2`       | `-0.01em`      | Sub-heroes                  |
| `headline-lg`  | `2rem`   | `1.25`      | `0`            | Page titles                 |
| `headline-md`  | `1.75rem`| `1.3`       | `0`            | Card titles                 |
| `headline-sm`  | `1.5rem` | `1.35`      | `0`            | Section titles              |
| `title-lg`     | `1.375rem`| `1.4`      | `0`            | Component titles            |
| `title-md`     | `1rem`   | `1.5`       | `0.01em`       | Sub-titles                  |
| `title-sm`     | `0.875rem`| `1.4`      | `0.01em`       | Minor titles                |
| `body-lg`      | `1rem`   | `1.6`       | `0.01em`       | Primary body copy           |
| `body-md`      | `0.875rem`| `1.5`      | `0.02em`       | Standard body               |
| `body-sm`      | `0.75rem`| `1.4`       | `0.02em`       | Captions, fine print        |
| `label-lg`     | `0.875rem`| `1.4`      | `0.04em`       | Button text, labels         |
| `label-md`     | `0.75rem`| `1.3`       | `0.1em`        | Metadata, HUD labels        |
| `label-sm`     | `0.6875rem`| `1.2`    | `0.1em`        | Micro-labels (e.g. "UNLOCK AT 60%") |

> [!IMPORTANT]
> Use wide letter-spacing (`0.1em`) for `label-sm` and `label-md` to emulate aerospace instrumentation. These labels should always be UPPERCASE.

---

## 3. Spacing & Layout

| Scale | Value   | Usage                          |
|-------|---------|--------------------------------|
| `xs`  | `4px`   | Tight internal padding         |
| `sm`  | `8px`   | Inline spacing, icon gaps      |
| `md`  | `16px`  | Standard component padding     |
| `lg`  | `24px`  | Section padding, card gaps     |
| `xl`  | `32px`  | Major section separators       |
| `2xl` | `48px`  | Page-level margins             |
| `3xl` | `64px`  | Hero spacing                   |
| `4xl` | `96px`  | Display-level breathing room   |

### Grid

- **Mobile:** Single column, `16px` gutter
- **Tablet:** 2 column fluid, `24px` gutter
- **Desktop:** 12 column, `32px` gutter, max-width `1440px`

---

## 4. Elevation & Depth

In "The Cinematic Astral," depth is an atmospheric effect, not just a shadow.

### Layering Principle

Stack `surface-container-lowest` cards on `surface-container-low` sections. This creates a natural "trough" or "lift" without visual clutter.

### Shadow System: "Bloom Shadows"

> [!IMPORTANT]
> Never use standard black drop shadows. If an element floats, it must **glow** or **blur** the background.

| Level      | CSS                                                                  | Usage                    |
|------------|----------------------------------------------------------------------|--------------------------|
| `bloom-sm` | `0 4px 20px rgba(0, 210, 255, 0.06)`                                | Subtle primary lift       |
| `bloom-md` | `0 8px 40px rgba(0, 210, 255, 0.10)`                                | Cards, floating panels    |
| `bloom-lg` | `0 16px 60px rgba(0, 210, 255, 0.15)`                               | Hero cards, modals        |
| `bloom-gold`| `0 8px 40px rgba(255, 215, 9, 0.10)`                               | Secondary/gold tier       |
| `bloom-red` | `0 8px 40px rgba(255, 61, 0, 0.10)`                                | Tertiary/red tier         |

### Ghost Border

If a container requires definition, use `outline-variant` at **15% opacity**. This should feel like a faint laser line, not a box:

```css
border: 1px solid rgba(69, 72, 79, 0.15);
```

### Glow Borders (Tier Cards)

For tier-specific cards, use a 2px stroke with a linear gradient of the tier accent color paired with an outer glow:

```css
/* Primary tier glow border */
border: 2px solid transparent;
border-image: linear-gradient(135deg, #81ecff, #00d4ec) 1;
box-shadow: 0 0 20px rgba(0, 210, 255, 0.2);
```

---

## 5. Glassmorphism

All primary panels must use `backdrop-blur` combined with semi-transparent surface colors.

```css
.glass-panel {
  background: rgba(34, 38, 47, 0.40);  /* surface_variant @ 40% */
  backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);
  border: 1px solid rgba(69, 72, 79, 0.15);  /* ghost border */
  border-radius: 16px;
}
```

### Glass Variants

| Variant       | Background Opacity | Blur    | Usage            |
|---------------|-------------------|---------|------------------|
| `glass-light` | 20%               | `16px`  | Overlays, tooltips|
| `glass-medium`| 40%               | `24px`  | Cards, panels     |
| `glass-heavy` | 60%               | `40px`  | Modals, drawers   |

---

## 6. Corner Roundness

| Token        | Value    | Usage                      |
|--------------|----------|----------------------------|
| `round-sm`   | `4px`    | Chips, tags                |
| `round-md`   | `8px`    | **Default** — buttons, inputs |
| `round-lg`   | `12px`   | Cards, containers          |
| `round-xl`   | `16px`   | Large cards, panels        |
| `round-2xl`  | `24px`   | Hero sections, modals      |
| `round-full` | `9999px` | Pill buttons, avatars      |

---

## 7. Components

### Buttons

#### Primary CTA (Cinematic)
- Shape: Pill (`round-full`)
- Background: gradient from `--primary` → `--primary-dim`
- Text: `--on-primary-fixed` (#003840)
- Shadow: `bloom-sm`
- Hover: increased glow intensity, subtle scale `1.02`

```css
.btn-primary {
  background: linear-gradient(135deg, #81ecff, #00d4ec);
  color: #003840;
  border-radius: 9999px;
  padding: 12px 32px;
  font-family: 'Plus Jakarta Sans', sans-serif;
  font-weight: 600;
  letter-spacing: 0.04em;
  box-shadow: 0 4px 20px rgba(0, 210, 255, 0.15);
  transition: all 0.3s ease;
}
.btn-primary:hover {
  box-shadow: 0 8px 40px rgba(0, 210, 255, 0.25);
  transform: scale(1.02);
}
```

#### Secondary CTA (Glass)
- Transparent background with ghost border and `backdrop-blur`
- Text: `--on-surface`
- Hover: border glow intensifies

```css
.btn-secondary {
  background: rgba(34, 38, 47, 0.3);
  backdrop-filter: blur(16px);
  border: 1px solid rgba(69, 72, 79, 0.15);
  color: #ecedf6;
  border-radius: 9999px;
  padding: 12px 32px;
  transition: all 0.3s ease;
}
.btn-secondary:hover {
  border-color: rgba(129, 236, 255, 0.4);
  box-shadow: 0 0 20px rgba(0, 210, 255, 0.1);
}
```

### Cards

#### Glass Card
```css
.card {
  background: rgba(34, 38, 47, 0.40);
  backdrop-filter: blur(24px);
  border: 1px solid rgba(69, 72, 79, 0.15);
  border-radius: 16px;
  padding: 24px;
  box-shadow: 0 8px 40px rgba(0, 210, 255, 0.06);
}
```

#### Tier Cards
- Large vertical containers with `round-xl` corners
- Feature a "Bloom Shadow" corresponding to tier color
- Overlapping thumbnails should use slight offsets and varied z-index stacking

### Progress Indicators ("The Journey Bar")

```css
.progress-track {
  background: #22262f;  /* surface_container_highest */
  border-radius: 9999px;
  height: 6px;
  overflow: visible;
}
.progress-fill {
  background: linear-gradient(90deg, #efc900, #ffd709);
  border-radius: 9999px;
  height: 100%;
  position: relative;
  box-shadow: 0 0 12px rgba(255, 215, 9, 0.4);
}
.progress-fill::after {
  content: '';
  position: absolute;
  right: -4px;
  top: 50%;
  transform: translateY(-50%);
  width: 12px;
  height: 12px;
  background: #ffd709;
  border-radius: 50%;
  box-shadow: 0 0 16px rgba(255, 215, 9, 0.6);
}
```

---

## 8. Motion & Animation

### Principles
- **Purposeful:** Every animation communicates state change
- **Smooth:** Use ease-out curves for entries, ease-in for exits
- **Subtle:** Micro-animations enhance; they never distract

### Timing

| Token           | Duration | Easing                  | Usage                 |
|-----------------|----------|-------------------------|-----------------------|
| `motion-fast`   | `150ms`  | `ease-out`              | Hovers, toggles       |
| `motion-normal` | `300ms`  | `cubic-bezier(0.4,0,0.2,1)` | Transitions, reveals |
| `motion-slow`   | `500ms`  | `cubic-bezier(0.4,0,0.2,1)` | Page transitions      |
| `motion-enter`  | `400ms`  | `cubic-bezier(0,0,0.2,1)`   | Elements entering     |
| `motion-exit`   | `200ms`  | `cubic-bezier(0.4,0,1,1)`   | Elements leaving      |

### Signature Animations

```css
/* Fade up entrance */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(20px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* Neon pulse for active elements */
@keyframes neonPulse {
  0%, 100% { box-shadow: 0 0 20px rgba(0, 210, 255, 0.15); }
  50%      { box-shadow: 0 0 40px rgba(0, 210, 255, 0.30); }
}

/* Glow shimmer for loading states */
@keyframes glowShimmer {
  0%   { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}
```

---

## 9. Iconography & Assets

- **Style:** Outlined / linear, 1.5px stroke
- **Size tokens:** `16px`, `20px`, `24px`, `32px`
- **Color:** Match `--on-surface-variant` by default; use accent tokens for interactive states
- **Recommendation:** Material Symbols (Outlined), Phosphor Icons, or Lucide

---

## 10. Design Rules

### ✅ DO

- **DO** use overlapping elements — let thumbnails bleed over glass panel edges for 3D depth
- **DO** use varying opacities — the background nebula should be partially visible through UI layers
- **DO** treat labels as HUD elements — `label-sm` with `0.1em` letter-spacing in UPPERCASE for metadata like "UNLOCK AT 60%"
- **DO** use radial gradients on active states to mimic glowing light sources
- **DO** use `surface-container` tiers for structural hierarchy instead of borders

### ❌ DON'T

- **DON'T** use 100% opaque black or white — pure colors break cinematic immersion
- **DON'T** use standard drop shadows — if it floats, it must glow or blur
- **DON'T** use sharp corners — stick to `round-md` (`8px`) minimum
- **DON'T** use 1px solid opaque borders for sectioning — use background shifts or ghost borders instead
- **DON'T** align thumbnails to strict grids — use slight offsets for a 3D collage effect

---

## 11. CSS Custom Properties Reference

```css
:root {
  /* Foundation */
  --surface: #0b0e14;
  --surface-dim: #0b0e14;
  --surface-bright: #282c36;
  --surface-container-lowest: #000000;
  --surface-container-low: #10131a;
  --surface-container: #161a21;
  --surface-container-high: #1c2028;
  --surface-container-highest: #22262f;
  --surface-variant: #22262f;
  --surface-tint: #81ecff;

  /* Primary */
  --primary: #81ecff;
  --primary-dim: #00d4ec;
  --primary-container: #00e3fd;
  --on-primary: #005762;
  --on-primary-container: #004d57;
  --on-primary-fixed: #003840;

  /* Secondary */
  --secondary: #ffd709;
  --secondary-dim: #efc900;
  --secondary-container: #705d00;
  --on-secondary: #5b4b00;
  --on-secondary-container: #fff7e6;

  /* Tertiary */
  --tertiary: #ff7350;
  --tertiary-dim: #dc3300;
  --tertiary-container: #fc3c00;
  --on-tertiary: #440900;
  --on-tertiary-container: #000000;

  /* Text */
  --on-background: #ecedf6;
  --on-surface: #ecedf6;
  --on-surface-variant: #a9abb3;

  /* Error */
  --error: #ff716c;
  --error-dim: #d7383b;
  --error-container: #9f0519;
  --on-error: #490006;
  --on-error-container: #ffa8a3;

  /* Structure */
  --outline: #73757d;
  --outline-variant: #45484f;
  --inverse-surface: #f9f9ff;
  --inverse-on-surface: #52555c;
  --inverse-primary: #006976;

  /* Typography */
  --font-display: 'Space Grotesk', sans-serif;
  --font-body: 'Plus Jakarta Sans', sans-serif;

  /* Roundness */
  --round-sm: 4px;
  --round-md: 8px;
  --round-lg: 12px;
  --round-xl: 16px;
  --round-2xl: 24px;
  --round-full: 9999px;

  /* Spacing */
  --space-xs: 4px;
  --space-sm: 8px;
  --space-md: 16px;
  --space-lg: 24px;
  --space-xl: 32px;
  --space-2xl: 48px;
  --space-3xl: 64px;
  --space-4xl: 96px;

  /* Motion */
  --motion-fast: 150ms ease-out;
  --motion-normal: 300ms cubic-bezier(0.4, 0, 0.2, 1);
  --motion-slow: 500ms cubic-bezier(0.4, 0, 0.2, 1);
}
```

---

## 12. Dark Mode Notes

This design system is **dark-mode native**. The entire palette is calibrated for dark backgrounds.

- **Color mode:** `DARK`
- **Appearance:** The deep space canvas (`#0b0e14`) is the foundation
- Light mode is not in scope — all surfaces, text, and accents are tuned for dark environments
- If a light mode is ever needed, use `--inverse-surface` (`#f9f9ff`) and `--inverse-on-surface` (`#52555c`) as a starting point

---

*Design system derived from the "Nebula Cinematic" theme — Stitch project: Cinematic Dashboard Redesign V1*
