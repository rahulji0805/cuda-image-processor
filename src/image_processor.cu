/**
 * image_processor.cu
 * CUDA-accelerated batch image processing: Grayscale conversion + Sobel edge detection
 *
 * Author: Rahul
 * Course: CUDA at Scale for the Enterprise (Coursera)
 *
 * Usage:
 *   ./image_processor -i <input_dir> -o <output_dir> -k <kernel_type> [-v]
 *   kernel_type: grayscale | edges | both
 */

#include "image_processor.h"

// ---------------------------------------------------------------------------
// CUDA Kernels
// ---------------------------------------------------------------------------

/**
 * Kernel 1: RGB -> Grayscale (luminance formula)
 * Each thread handles one pixel.
 */
__global__ void rgbToGrayscaleKernel(const unsigned char *d_rgb,
                                      unsigned char *d_gray,
                                      int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height)
    {
        int grayIdx = row * width + col;
        int rgbIdx  = grayIdx * 3;

        unsigned char r = d_rgb[rgbIdx + 0];
        unsigned char g = d_rgb[rgbIdx + 1];
        unsigned char b = d_rgb[rgbIdx + 2];

        // ITU-R BT.601 luminance
        d_gray[grayIdx] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

/**
 * Kernel 2: Sobel edge detection on grayscale image.
 * Uses shared memory to cache a tile + 1-pixel halo for efficient access.
 */
__global__ void sobelEdgeKernel(const unsigned char *d_gray,
                                 unsigned char *d_edges,
                                 int width, int height)
{
    // Shared memory tile: (TILE+2) x (TILE+2) to include halo
    __shared__ unsigned char smem[TILE_DIM + 2][TILE_DIM + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global pixel coordinates (center of halo)
    int col = blockIdx.x * TILE_DIM + tx - 1;
    int row = blockIdx.y * TILE_DIM + ty - 1;

    // Clamp to image boundary (replicate border)
    int clamped_col = max(0, min(col, width  - 1));
    int clamped_row = max(0, min(row, height - 1));

    smem[ty][tx] = d_gray[clamped_row * width + clamped_col];
    __syncthreads();

    // Only interior threads compute output (skip halo threads)
    if (tx > 0 && tx < TILE_DIM + 1 && ty > 0 && ty < TILE_DIM + 1)
    {
        int out_col = blockIdx.x * TILE_DIM + (tx - 1);
        int out_row = blockIdx.y * TILE_DIM + (ty - 1);

        if (out_col < width && out_row < height)
        {
            // Sobel Gx
            int gx = -1 * smem[ty-1][tx-1] + 1 * smem[ty-1][tx+1]
                   + -2 * smem[ty  ][tx-1] + 2 * smem[ty  ][tx+1]
                   + -1 * smem[ty+1][tx-1] + 1 * smem[ty+1][tx+1];

            // Sobel Gy
            int gy = -1 * smem[ty-1][tx-1] + -2 * smem[ty-1][tx]  + -1 * smem[ty-1][tx+1]
                   +  1 * smem[ty+1][tx-1] +  2 * smem[ty+1][tx]  +  1 * smem[ty+1][tx+1];

            int magnitude = (int)sqrtf((float)(gx*gx + gy*gy));
            d_edges[out_row * width + out_col] = (unsigned char)min(magnitude, 255);
        }
    }
}

// ---------------------------------------------------------------------------
// Host helper functions
// ---------------------------------------------------------------------------

__host__ void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error [%s]: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/**
 * Process a single image: allocate device memory, run kernels, copy back.
 * Returns elapsed time in milliseconds.
 */
__host__ float processImage(const unsigned char *h_rgb,
                             unsigned char *h_gray,
                             unsigned char *h_edges,
                             int width, int height,
                             const std::string &kernelType)
{
    int numPixels   = width * height;
    size_t rgbSize  = numPixels * 3 * sizeof(unsigned char);
    size_t graySize = numPixels     * sizeof(unsigned char);

    unsigned char *d_rgb = nullptr, *d_gray = nullptr, *d_edges = nullptr;

    checkCudaError(cudaMalloc(&d_rgb,   rgbSize),  "malloc d_rgb");
    checkCudaError(cudaMalloc(&d_gray,  graySize), "malloc d_gray");
    checkCudaError(cudaMalloc(&d_edges, graySize), "malloc d_edges");

    checkCudaError(cudaMemcpy(d_rgb, h_rgb, rgbSize, cudaMemcpyHostToDevice), "memcpy H->D rgb");

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Grid/block config
    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim((width  + BLOCK_DIM - 1) / BLOCK_DIM,
                 (height + BLOCK_DIM - 1) / BLOCK_DIM);

    // --- Grayscale kernel ---
    if (kernelType == "grayscale" || kernelType == "both")
    {
        rgbToGrayscaleKernel<<<gridDim, blockDim>>>(d_rgb, d_gray, width, height);
        checkCudaError(cudaGetLastError(), "grayscale kernel launch");
    }

    // --- Sobel kernel (needs grayscale first) ---
    if (kernelType == "edges" || kernelType == "both")
    {
        if (kernelType == "edges")
        {
            // If only edges requested, still need grayscale as intermediate
            rgbToGrayscaleKernel<<<gridDim, blockDim>>>(d_rgb, d_gray, width, height);
            checkCudaError(cudaGetLastError(), "grayscale (pre-sobel) kernel launch");
            cudaDeviceSynchronize();
        }

        dim3 sobelBlock(TILE_DIM + 2, TILE_DIM + 2);
        dim3 sobelGrid((width  + TILE_DIM - 1) / TILE_DIM,
                       (height + TILE_DIM - 1) / TILE_DIM);
        sobelEdgeKernel<<<sobelGrid, sobelBlock>>>(d_gray, d_edges, width, height);
        checkCudaError(cudaGetLastError(), "sobel kernel launch");
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedMs = 0.0f;
    cudaEventElapsedTime(&elapsedMs, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy results back
    if (kernelType == "grayscale" || kernelType == "both")
        checkCudaError(cudaMemcpy(h_gray, d_gray, graySize, cudaMemcpyDeviceToHost), "memcpy D->H gray");
    if (kernelType == "edges" || kernelType == "both")
        checkCudaError(cudaMemcpy(h_edges, d_edges, graySize, cudaMemcpyDeviceToHost), "memcpy D->H edges");

    cudaFree(d_rgb);
    cudaFree(d_gray);
    cudaFree(d_edges);

    return elapsedMs;
}

// ---------------------------------------------------------------------------
// PGM / PPM minimal I/O (no external lib dependency)
// ---------------------------------------------------------------------------

__host__ bool loadPPM(const std::string &filename,
                       unsigned char **data, int &width, int &height)
{
    FILE *fp = fopen(filename.c_str(), "rb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", filename.c_str()); return false; }

    char magic[3];
    fscanf(fp, "%2s", magic);
    if (strcmp(magic, "P6") != 0) { fclose(fp); return false; }

    // Skip comments
    int c = fgetc(fp);
    while (c == '#') { while (fgetc(fp) != '\n'); c = fgetc(fp); }
    ungetc(c, fp);

    int maxVal;
    fscanf(fp, "%d %d %d", &width, &height, &maxVal);
    fgetc(fp); // consume newline

    *data = (unsigned char *)malloc(width * height * 3);
    fread(*data, 1, width * height * 3, fp);
    fclose(fp);
    return true;
}

__host__ void savePGM(const std::string &filename,
                       const unsigned char *data, int width, int height)
{
    FILE *fp = fopen(filename.c_str(), "wb");
    fprintf(fp, "P5\n%d %d\n255\n", width, height);
    fwrite(data, 1, width * height, fp);
    fclose(fp);
}

// ---------------------------------------------------------------------------
// CLI argument parsing
// ---------------------------------------------------------------------------

__host__ AppConfig parseArgs(int argc, char *argv[])
{
    AppConfig cfg;
    cfg.inputDir   = "data/input";
    cfg.outputDir  = "data/output";
    cfg.kernelType = "both";
    cfg.verbose    = false;

    for (int i = 1; i < argc; i++)
    {
        std::string opt(argv[i]);
        if (opt == "-i" && i + 1 < argc) { cfg.inputDir   = argv[++i]; }
        else if (opt == "-o" && i + 1 < argc) { cfg.outputDir  = argv[++i]; }
        else if (opt == "-k" && i + 1 < argc) { cfg.kernelType = argv[++i]; }
        else if (opt == "-v") { cfg.verbose = true; }
        else if (opt == "-h") {
            printf("Usage: ./image_processor -i <input_dir> -o <output_dir> -k <grayscale|edges|both> [-v]\n");
            exit(0);
        }
    }
    return cfg;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    AppConfig cfg = parseArgs(argc, argv);

    printf("=== CUDA Batch Image Processor ===\n");
    printf("Input  : %s\n", cfg.inputDir.c_str());
    printf("Output : %s\n", cfg.outputDir.c_str());
    printf("Kernel : %s\n", cfg.kernelType.c_str());

    // Create output dir
    std::string mkdirCmd = "mkdir -p " + cfg.outputDir;
    system(mkdirCmd.c_str());

    // List .ppm files in input dir
    std::vector<std::string> imageFiles;
    {
        std::string listCmd = "ls " + cfg.inputDir + "/*.ppm 2>/dev/null";
        FILE *pipe = popen(listCmd.c_str(), "r");
        char buf[512];
        while (fgets(buf, sizeof(buf), pipe))
        {
            std::string fname(buf);
            fname.erase(fname.find_last_not_of(" \n\r\t") + 1);
            imageFiles.push_back(fname);
        }
        pclose(pipe);
    }

    if (imageFiles.empty())
    {
        fprintf(stderr, "No .ppm files found in %s\n", cfg.inputDir.c_str());
        fprintf(stderr, "Run: python3 scripts/generate_test_images.py to create test data\n");
        return 1;
    }

    printf("Found %zu image(s) to process\n\n", imageFiles.size());

    // Results log
    FILE *logFp = fopen("results/processing_log.csv", "w");
    fprintf(logFp, "filename,width,height,kernel,elapsed_ms\n");

    double totalTime = 0.0;
    int processed = 0;

    for (const auto &filepath : imageFiles)
    {
        unsigned char *h_rgb = nullptr;
        int width = 0, height = 0;

        if (!loadPPM(filepath, &h_rgb, width, height))
        {
            fprintf(stderr, "Skipping %s (load failed)\n", filepath.c_str());
            continue;
        }

        int numPixels = width * height;
        unsigned char *h_gray  = (unsigned char *)malloc(numPixels);
        unsigned char *h_edges = (unsigned char *)malloc(numPixels);

        float elapsed = processImage(h_rgb, h_gray, h_edges, width, height, cfg.kernelType);
        totalTime += elapsed;

        // Extract base filename
        std::string base = filepath.substr(filepath.find_last_of("/\\") + 1);
        base = base.substr(0, base.find_last_of('.'));

        // Save outputs
        if (cfg.kernelType == "grayscale" || cfg.kernelType == "both")
        {
            std::string outPath = cfg.outputDir + "/" + base + "_gray.pgm";
            savePGM(outPath, h_gray, width, height);
        }
        if (cfg.kernelType == "edges" || cfg.kernelType == "both")
        {
            std::string outPath = cfg.outputDir + "/" + base + "_edges.pgm";
            savePGM(outPath, h_edges, width, height);
        }

        fprintf(logFp, "%s,%d,%d,%s,%.4f\n",
                base.c_str(), width, height, cfg.kernelType.c_str(), elapsed);

        if (cfg.verbose)
            printf("[%d] %s (%dx%d) -> %.4f ms\n", processed+1, base.c_str(), width, height, elapsed);

        free(h_rgb);
        free(h_gray);
        free(h_edges);
        processed++;
    }

    fclose(logFp);

    printf("\n=== Results ===\n");
    printf("Images processed : %d\n", processed);
    printf("Total GPU time   : %.4f ms\n", totalTime);
    printf("Avg per image    : %.4f ms\n", processed > 0 ? totalTime / processed : 0.0);
    printf("Log saved to     : results/processing_log.csv\n");

    return 0;
}
