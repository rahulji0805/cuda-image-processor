#!/usr/bin/env bash
set -e

echo "=== CUDA Batch Image Processor - run.sh ==="

# Step 1: Build
make clean build

# Step 2: Generate test images if not present
if [ ! "$(ls -A data/input/*.ppm 2>/dev/null)" ]; then
    echo "Generating test images..."
    python3 scripts/generate_test_images.py
fi

mkdir -p results data/output

# Step 3: Run all kernel types
for kernel in grayscale edges both
do
    echo ""
    echo "--- Running kernel: $kernel ---"
    make run ARGS="-i data/input -o data/output -k $kernel -v"
done

# Step 4: Generate summary plot
echo ""
echo "--- Generating performance plot ---"
python3 scripts/plot_results.py

echo ""
echo "Done. Check data/output/ for processed images and results/ for logs/plots."
