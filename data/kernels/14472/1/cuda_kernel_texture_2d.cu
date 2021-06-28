#include "includes.h"
__global__ void cuda_kernel_texture_2d(unsigned char *surface, int width, int height, size_t pitch, float t) {
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;

// in the case where, due to quantization into grids, we have
// more threads than pixels, skip the threads which don't
// correspond to valid pixels
if (x >= width || y >= height) {
return;
}

// get a pointer to the pixel at (x,y)
float *pixel = (float *)(surface + y * pitch) + 4 * x;

// populate it
float value_x = 0.5f + 0.5f * cos(t + 10.0f * ((2.0f * x) / width  - 1.0f));
float value_y = 0.5f + 0.5f * cos(t + 10.0f * ((2.0f * y) / height - 1.0f));

pixel[0] = 0.5 * pixel[0] + 0.5 * pow(value_x, 3.0f); // red
pixel[1] = 0.5 * pixel[1] + 0.5 * pow(value_y, 3.0f); // green
pixel[2] = 0.5f + 0.5f * cos(t); // blue
pixel[3] = 1.0f; // alpha
}