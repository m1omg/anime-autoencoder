# 🧭🧠 Anime autoencoder — latent space explorer

A **from-scratch autoencoder** (vanilla JS, no libraries) that trains **live in your browser**
to compress anime faces down to just **two numbers** — and decompress them back.

**▶ Live:** https://m1omg.github.io/anime-autoencoder/

Drag the cursor around the 2D latent map and the decoder renders, in real time, whatever
"lives" at that point — including everything *between* the images it learned. The mosaic in
the background is the decoded latent plane itself, so you can literally see the map the
network built.

## ⚡ WebGL version
[`webgl.html`](https://m1omg.github.io/anime-autoencoder/webgl.html) trains the same autoencoder
**entirely on the GPU**: every matrix multiply, the full backpropagation and the Adam optimizer run
as WebGL2 fragment-shader passes over `R32F` float textures (still zero dependencies). The extra
horsepower is spent on quality:
- **Bigger network** — `6912 → 256 → 64 → 2 → 64 → 256 → 6912` (~3.6M parameters vs ~1.8M on the CPU
  version) with **batch 32** instead of 8, for smoother gradients and sharper reconstructions.
- **16×16 mosaic** (vs 9×9) decoded on the GPU in batched passes, fully refreshed every 4 frames
  instead of trickling in 2 cells per frame, with bilinear filtering in the shader.
- **Bicubic (Catmull-Rom) upscaling** of the decoder preview and **HiDPI/retina-aware** canvases.
- **Adaptive scheduling** — one tiny loss readback per frame doubles as a GPU sync point, and the
  steps-per-frame count self-tunes to your training time budget slider.

If WebGL2 float rendering isn't available the page points you back to the CPU version.

## How it works
- **Architecture:** `N → 128 → 32 → 2 → 32 → 128 → N` where `N = width·height·3`
  (default 64×64 RGB = 12,288, selectable 48 / 64 / 96), ReLU hidden layers, sigmoid output,
  trained with backpropagation + Adam — all hand-written on flat `Float32Array`s, zero dependencies.
- **Resolution selector (48 / 64 / 96):** since this is a pixel-vector autoencoder, input and
  output resolution are locked together and cost grows *quadratically*, so it's a live control.
  To keep it from melting a phone/tablet, the minibatch shrinks as resolution rises (so per-step
  cost — and frame pacing — stays roughly constant) and per-frame decode work scales down too,
  keeping cursor-dragging smooth even at 96².
- **Three compute backends (toggleable live):**
  - **🧮 CPU** — training runs on the main thread (simple, always available).
  - **🧵 Web Worker** (default) — the exact same trainer runs in a background thread, so the UI
    stays at ~60 fps no matter how hard it's training — and because frame pacing no longer
    depends on it, it always uses the full batch size (CPU mode must shrink the batch at high
    resolutions to stay responsive).
  - **⚡ WebGPU** — forward, backprop, and Adam run as WGSL compute shaders on the GPU. It ships
    with a **forward self-test against the CPU engine** plus a **live loss-divergence guard**:
    if the GPU is unavailable or produces incorrect results, it falls back to CPU automatically
    and says so (with the measured mismatch in the status line). Tolerances account for
    legitimate float32/FMA drift across GPU vendors — a wrong kernel is orders of magnitude
    outside them. (All three share one rendering path: whichever backend trains, it periodically
    hands a weight snapshot back to the main thread, which does all the drawing.)
- **Linear latent + center-pull regularizer:** a tanh bottleneck saturates during the early
  chaotic phase of training and permanently freezes the encoder (a fun failure mode found
  while building this). A linear 2-unit latent with a gentle L2 pull toward the origin — plus
  a short LR warmup — fixes it.
- **✨ Sharpness (gradient) loss:** alongside pixel MSE, the decoder is trained to match the
  *spatial gradients* (edges) of the target, not just its pixels. Plain MSE rewards averaging
  between plausible options, which is exactly why autoencoders blur — the gradient term pushes
  back toward crisp edges. It's a toggle, so you can flip it off and watch reconstructions turn
  to mush.
- **Latent noise injection** during training keeps the space smooth between clusters, so
  dragging morphs instead of jumping. Both the noise and the center pull **anneal to 25%**
  as training progresses — kept at full strength forever, they glue each class's augmented
  variants onto a single latent point and the decoder can only paint their blurred average.
- **Augmentations:** each source image becomes 16 variants (shift / zoom / rotate / mirror),
  so real clusters form in the latent space instead of lonely isolated points.
- **Eight faces** populate the space, so interpolation traverses a genuinely varied set instead
  of cross-fading a handful of attractors. Note the honest limit: two latent numbers can't hold
  eight faces *sharply*, so individual reconstructions stay soft — that 2-D bottleneck, not the
  loss, is the hard wall. Interpolation ≠ compositional part-swapping.
- **Live views:** decoded mosaic of the whole plane (progressively refreshed), training
  samples as live-encoded colored dots, adaptive map range, log-scale loss curve, PSNR.
- **🎬 Tour mode** animates the cursor between class centroids; you can also **add your own
  image** as a new class mid-training and watch the map reorganize.

## Run locally
Must be served over HTTP (the browser needs to read pixel data), not opened as `file://`:
```bash
python3 -m http.server
# then open http://localhost:8000/
```

Sister project of [neural-oliver](https://github.com/m1omg/neural-oliver), where a coordinate
MLP (SIREN, Fourier features & friends) learns to paint a single image.

Characters: Klee & Sigewinne (Genshin Impact), Cirno & Flandre Scarlet & Reimu Hakurei
(Touhou Project), Konata Izumi (Lucky Star), Megumin (KonoSuba), and Kanna Kamui (Miss
Kobayashi's Dragon Maid) — safe-rated fan-art images, used here for a non-commercial
educational demo.
