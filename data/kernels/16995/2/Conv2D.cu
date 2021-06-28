#include "includes.h"

# define MAX(a, b) ((a) > (b) ? (a) : (b))

# define GAUSSIAN_KERNEL_SIZE 3
# define SOBEL_KERNEL_SIZE 5
# define TILE_WIDTH 32
# define SMEM_SIZE 128
__global__ void Conv2D(float *d_image, float *kernel, float *d_result, int width, int height, int kernelSize) {
const int sharedMemWidth = TILE_WIDTH + MAX(SOBEL_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE) - 1;
__shared__ float sharedMem[sharedMemWidth][sharedMemWidth];

int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
int destY = dest / sharedMemWidth;
int destX = dest % sharedMemWidth;
int srcY = blockIdx.y * TILE_WIDTH + destY - (kernelSize / 2);
int srcX = blockIdx.x * TILE_WIDTH + destX - (kernelSize / 2);
int src = (srcY * width + srcX);
if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
sharedMem[destY][destX] = d_image[src];
else
sharedMem[destY][destX] = 0;

dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
destY = dest / sharedMemWidth;
destX = dest % sharedMemWidth;
srcY = blockIdx.y * TILE_WIDTH + destY - (kernelSize / 2);
srcX = blockIdx.x * TILE_WIDTH + destX - (kernelSize / 2);
src = (srcY * width + srcX);
if (destY < sharedMemWidth) {
if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
sharedMem[destY][destX] = d_image[src];
else
sharedMem[destY][destX] = 0;
}
__syncthreads();

float accum = 0;
for (int j = 0; j < kernelSize; j++)
for (int i = 0; i < kernelSize; i++)
accum += sharedMem[threadIdx.y + j][threadIdx.x + i] * kernel[j * kernelSize + i];
int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
int y = blockIdx.y * TILE_WIDTH + threadIdx.y;
if (x < width && y < height)
d_result[y * width + x] = (fminf(fmaxf((accum), 0.0), 1.0));
__syncthreads();
}