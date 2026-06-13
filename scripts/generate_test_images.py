#!/usr/bin/env python3
"""
generate_test_images.py
Creates a set of synthetic PPM test images of varying sizes for the CUDA processor.
"""
import os
import struct
import random
import math

OUTPUT_DIR = "data/input"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_ppm(filename, pixels, width, height):
    """Save RGB pixel array as binary PPM (P6)."""
    with open(filename, 'wb') as f:
        header = f"P6\n{width} {height}\n255\n"
        f.write(header.encode('ascii'))
        f.write(bytes(pixels))

def gen_gradient(width, height):
    pixels = []
    for y in range(height):
        for x in range(width):
            r = int(255 * x / width)
            g = int(255 * y / height)
            b = 128
            pixels += [r, g, b]
    return pixels

def gen_checkerboard(width, height, tile=32):
    pixels = []
    for y in range(height):
        for x in range(width):
            if (x // tile + y // tile) % 2 == 0:
                pixels += [255, 255, 255]
            else:
                pixels += [30, 30, 30]
    return pixels

def gen_circles(width, height, n=8):
    pixels = [100, 100, 100] * width * height  # gray background
    cx, cy = width // 2, height // 2
    for i in range(n):
        angle = 2 * math.pi * i / n
        ox = int(cx + (width // 4) * math.cos(angle))
        oy = int(cy + (height // 4) * math.sin(angle))
        r_color = random.randint(50, 255)
        g_color = random.randint(50, 255)
        b_color = random.randint(50, 255)
        radius = min(width, height) // 8
        for dy in range(-radius, radius):
            for dx in range(-radius, radius):
                if dx*dx + dy*dy <= radius*radius:
                    px, py = ox + dx, oy + dy
                    if 0 <= px < width and 0 <= py < height:
                        idx = (py * width + px) * 3
                        pixels[idx]   = r_color
                        pixels[idx+1] = g_color
                        pixels[idx+2] = b_color
    return pixels

def gen_noise(width, height):
    return [random.randint(0, 255) for _ in range(width * height * 3)]

def gen_stripes(width, height, stripe_w=20):
    pixels = []
    colors = [(220,50,50),(50,220,50),(50,50,220),(220,220,50),(50,220,220)]
    for y in range(height):
        for x in range(width):
            c = colors[(x // stripe_w) % len(colors)]
            pixels += list(c)
    return pixels

# Generate images of different sizes
configs = [
    ("small_gradient_64",    64,   64,  gen_gradient),
    ("small_checkerboard_64",64,   64,  gen_checkerboard),
    ("medium_gradient_256",  256,  256, gen_gradient),
    ("medium_circles_256",   256,  256, gen_circles),
    ("medium_checkerboard_256", 256, 256, gen_checkerboard),
    ("medium_stripes_256",   256,  256, gen_stripes),
    ("large_gradient_512",   512,  512, gen_gradient),
    ("large_checkerboard_512",512, 512, gen_checkerboard),
    ("large_circles_512",    512,  512, gen_circles),
    ("large_noise_512",      512,  512, gen_noise),
    ("xlarge_gradient_1024", 1024, 1024,gen_gradient),
    ("xlarge_circles_1024",  1024, 1024,gen_circles),
]

print(f"Generating {len(configs)} test images in '{OUTPUT_DIR}/'...")
for name, w, h, fn in configs:
    pixels = fn(w, h)
    path = os.path.join(OUTPUT_DIR, f"{name}.ppm")
    save_ppm(path, pixels, w, h)
    print(f"  {name}.ppm  ({w}x{h})")

print(f"\nDone. {len(configs)} images created.")
