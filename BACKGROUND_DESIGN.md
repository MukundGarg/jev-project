# Background Design: WakeGlyph (ASCII Pointer-Wake Field)

Status: planned, not yet implemented. This document records the design and the integration plan.

## 1. Concept

A full-bleed background made of monospace characters on a canvas. At rest it shows a sparse, faint, static scatter of glyphs. When the pointer moves, it leaves a "wake" of brighter, denser glyphs that decays back to the resting scatter.

The wake responds to pointer speed:

| Pointer motion | Wake shape | Lifetime |
|---|---|---|
| Fast | long, thin trail (small radius, many cells along the path) | short (fast decay) |
| Slow | fat blob (larger radius) | lingering (slow decay) |

Intensity is rendered by mapping each cell's value to a character from an ASCII ramp (`" .:-=+*#%@"`): low values draw light characters, high values draw dense ones, and alpha scales with the value.

## 2. How it works

- **Grid:** the canvas is divided into cells (`cellSize`, default 12px). Three `Float32Array`s of length cols x rows hold per-cell state: `heat` (current stamped value), `rate` (decay speed fixed at stamp time) and `ambient` (static resting value).
- **Ambient scatter:** about 3.5% of cells get a faint value from a seeded PRNG (mulberry32), so the resting pattern is stable within a session and across re-renders at the same size.
- **Stamping:** on `pointermove`, the pointer's speed (px/ms) sets the stamp radius (0.6 to 2.6 cells) and the decay rate (0.8 to 4.5). The path since the last event is sampled every half cell so fast movement leaves no gaps. A cell's heat is only overwritten if the new falloff value is higher.
- **Frame loop:** `requestAnimationFrame` decays all heated cells by `rate * dt` (dt clamped to 64ms), then redraws. Displayed value is `max(ambient, heat)`.
- **Resolution:** device pixel ratio capped at 2; resize is debounced (150ms) and rebuilds the grid.
- **Theming:** ink color is read from the canvas's computed CSS `color` (Tailwind `text-foreground`). A `MutationObserver` on `<html class>` refreshes it when the theme toggles, with no remount. Font comes from the `--font-mono` CSS variable, falling back to `ui-monospace, monospace`.
- **Accessibility:** canvas is `aria-hidden`. With `prefers-reduced-motion: reduce`, only a single static ambient frame is drawn and pointer listeners are not attached.

## 3. API

```tsx
<WakeGlyph cellSize={12} className="absolute inset-0" />
```

| Prop | Type | Default | Meaning |
|---|---|---|---|
| `cellSize` | number | 12 | grid cell size in px (smaller = denser and costlier) |
| `className` | string | "" | extra classes on the canvas |

No state management, context providers or data props are required. No image assets or icons are needed.

## 4. Requirements and dependencies

- React 18+ with TypeScript.
- Tailwind CSS with shadcn-style theme tokens (`text-foreground`, `bg-background`) and a `.dark` class on `<html>` for dark mode.
- Path alias `@/` mapped to the project root or `src/` (the demo imports `@/components/ui/background-ascii-wake`).
- `"use client"` at the top is for Next.js App Router and is harmless elsewhere.
- **No external npm packages.** Nothing to install beyond the project's existing React, TypeScript and Tailwind.
- Optional: define `--font-mono` in global CSS to control the glyph font.

If the target project does not already use shadcn, Tailwind and TypeScript: initialize with `npx shadcn@latest init` (this sets up Tailwind, the `@/` alias and the theme tokens). Components then live in `components/ui/`. That folder is the shadcn convention and the CLI's `components.json` points to it. Keeping the file there means later `shadcn add` commands and the `@/components/ui/...` imports resolve consistently.

## 5. Integration plan

1. Confirm the project is React + TypeScript + Tailwind + shadcn structure; set it up as above if not.
2. Create `components/ui/background-ascii-wake.tsx` with the component source (provided in the original brief).
3. Use the demo as a smoke test: full-screen `relative` container with `bg-background` and `<WakeGlyph className="absolute inset-0" />`.
4. Verify in light and dark theme, at several viewport sizes, and with reduced motion enabled in OS settings.
5. Place it behind the real page content (see section 6).

## 5b. Where to use it

Best as a hero or landing-section backdrop, or behind an empty state. Not suited as a persistent background behind dense reading content or forms.

## 6. Known gotchas and decisions

- **Pointer events:** the wake listens on the canvas itself. If page content sits on top of the canvas, it will block pointer events and the wake will not appear. Options: (a) keep the canvas in a region with no content above it, or (b) move the listeners to a parent or `window` (requires a small code change). Decide this before placing it behind real UI.
- **Touch devices:** no hover on touch; the wake will only appear while dragging, and scrolling gestures may not fire `pointermove` reliably. Treat the resting scatter as the touch experience.
- **Performance:** the rAF loop and a full canvas redraw run continuously, even when idle. Larger screens and small `cellSize` raise cost. Acceptable for a single hero section; consider pausing when off-screen (IntersectionObserver) or when the tab is hidden if used more broadly.
- **Changing `cellSize`** re-runs the effect and rebuilds the grid; do not animate it.
- **Reduced motion** is read once at mount; toggling the OS setting needs a reload.
- **Contrast:** glyph alpha tops out at 1 with the full foreground color. If text overlays the canvas, lower the canvas opacity via `className` (e.g. `opacity-40`) to protect readability.

## 7. Open questions

1. Which project is this background for? If it is the ContextSelect demo UI, note that the planning docs chose Gradio, which cannot host a React component. Using this would mean replacing Gradio with a small React frontend plus a Python API, which expands scope. If it is a separate site, no conflict.
2. Should the pointer listener stay on the canvas or move to the page level?
3. Is a static fallback (ambient only) acceptable on touch devices?

## 8. Acceptance criteria

- Renders full-bleed with no layout shift and no console errors.
- Wake is visibly thin and short-lived on fast movement and fat and lingering on slow movement.
- Resting scatter is sparse, stable and faint.
- Color follows the light/dark theme toggle without reload.
- With reduced motion enabled, only the static frame is drawn.
- Resizing the window rebuilds the grid correctly.
