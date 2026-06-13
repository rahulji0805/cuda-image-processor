# CUDA Batch Image Processor

GPU-accelerated batch image processing using CUDA — performs **grayscale conversion** and **Sobel edge detection** on hundreds of images in parallel.

Built for the *CUDA at Scale for the Enterprise* Coursera course (Independent Project).

---

## What It Does

| Kernel | Description | Memory Used |
|--------|-------------|-------------|
| `grayscale` | RGB → Grayscale via ITU-R BT.601 luminance formula | Global |
| `edges` | Sobel edge detection (Gx + Gy gradient magnitude) | Shared (tiled) |
| `both` | Grayscale + Edge detection in one pass | Global + Shared |

Each thread handles one pixel in the grayscale kernel. The Sobel kernel uses **shared memory tiling** with a 1-pixel halo to minimize global memory traffic.

---

## Project Structure

```
cuda_image_processor/
├── src/
│   ├── image_processor.cu    # CUDA kernels + host code
│   └── image_processor.h     # Header / constants
├── scripts/
│   ├── generate_test_images.py  # Creates synthetic PPM test data
│   └── plot_results.py          # Performance chart generator
├── data/
│   ├── input/    # Input .ppm images go here
│   └── output/   # Processed .pgm images saved here
├── results/      # Logs and charts
├── Makefile
├── run.sh
└── README.md
```

---

## Requirements

- CUDA Toolkit ≥ 10.0
- `nvcc` (tested on sm_50+)
- Python 3 (for test image generation + plotting)
- `matplotlib`, `numpy` (optional, for charts)

---

## Quick Start

```bash
# 1. Clone and enter
git clone <https://github.com/rahulji0805/cuda-image-processor>
cd cuda_image_processor

# 2. Generate test images (12 synthetic PPMs, 64px to 1024px)
python3 scripts/generate_test_images.py

# 3. Build
make

# 4. Run (processes all images with all kernels)
./run.sh

# OR run manually with options:
./image_processor -i data/input -o data/output -k both -v
```

---

## CLI Arguments

```
./image_processor -i <input_dir> -o <output_dir> -k <kernel_type> [-v]

  -i   Input directory containing .ppm files  (default: data/input)
  -o   Output directory for results           (default: data/output)
  -k   Kernel: grayscale | edges | both       (default: both)
  -v   Verbose output (print per-image stats)
  -h   Show help
```

---

## Performance (Sample Results)

| Kernel | Avg Time (ms) | Dataset |
|--------|--------------|---------|
| grayscale | ~0.05 | 12 images, 64-1024px |
| edges | ~0.12 | 12 images, 64-1024px |
| both | ~0.14 | 12 images, 64-1024px |

Times measured with `cudaEventRecord`. Includes only kernel execution (not H→D / D→H transfer).

---

## Key Implementation Details

### Grayscale Kernel
```cuda
// One thread per pixel, 2D grid
d_gray[idx] = 0.299f * R + 0.587f * G + 0.114f * B;
```

### Sobel Kernel (Shared Memory)
- Tile size: 16×16 with 1-pixel halo → 18×18 shared memory per block
- `__syncthreads()` ensures halo is loaded before computation
- Border handled by clamping (replicate padding)

---

## Output Files

- `data/output/<name>_gray.pgm` — Grayscale result
- `data/output/<name>_edges.pgm` — Edge-detected result  
- `results/processing_log.csv` — Per-image timing log
- `results/performance_chart.png` — Bar chart of avg times per kernel

---

## Lessons Learned

1. **Shared memory** tiling in the Sobel kernel gives measurable speedup over naive global memory access, especially for larger images where cache misses dominate.
2. **2D grid/block** configuration (`dim3`) maps naturally to image coordinates and simplifies index math.
3. **cudaEvent** timing is more accurate than CPU-side timing as it captures only device execution.
4. PPM/PGM format requires no external libraries, keeping the project self-contained on any CUDA-capable machine.
