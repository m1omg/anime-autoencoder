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
- **Architecture:** `6912 → 128 → 32 → 2 → 32 → 128 → 6912` (48×48 RGB in and out),
  ReLU hidden layers, sigmoid output, trained with backpropagation + Adam — all hand-written
  on flat `Float32Array`s, zero dependencies.
- **Linear latent + center-pull regularizer:** a tanh bottleneck saturates during the early
  chaotic phase of training and permanently freezes the encoder (a fun failure mode found
  while building this). A linear 2-unit latent with a gentle L2 pull toward the origin — plus
  a short LR warmup — fixes it.
- **Latent noise injection** during training keeps the space smooth between clusters, so
  dragging morphs instead of jumping.
- **Augmentations:** each source image becomes 16 variants (shift / zoom / rotate / mirror),
  so real clusters form in the latent space instead of three lonely points.
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

Characters: Klee (Genshin Impact), Cirno & Flandre Scarlet (Touhou Project) — fan-art images,
used here for a non-commercial educational demo.
