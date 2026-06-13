#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string>
#include <vector>
#include <cuda_runtime.h>

// Kernel tuning constants
#define BLOCK_DIM  16
#define TILE_DIM   16

struct AppConfig {
    std::string inputDir;
    std::string outputDir;
    std::string kernelType;  // grayscale | edges | both
    bool verbose;
};

// Kernels
__global__ void rgbToGrayscaleKernel(const unsigned char *d_rgb,
                                      unsigned char *d_gray,
                                      int width, int height);

__global__ void sobelEdgeKernel(const unsigned char *d_gray,
                                 unsigned char *d_edges,
                                 int width, int height);

// Host functions
__host__ void   checkCudaError(cudaError_t err, const char *msg);
__host__ float  processImage(const unsigned char *h_rgb,
                              unsigned char *h_gray, unsigned char *h_edges,
                              int width, int height, const std::string &kernelType);
__host__ bool   loadPPM(const std::string &filename,
                         unsigned char **data, int &width, int &height);
__host__ void   savePGM(const std::string &filename,
                         const unsigned char *data, int width, int height);
__host__ AppConfig parseArgs(int argc, char *argv[]);
