#include "includes.h"
__global__ void cudaComputeYGradient(int* y_gradient, unsigned char* channel, int image_width, int image_height, int chunk_size_per_thread) {
int y_kernel[3][3] = { { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } };
int index = blockIdx.x * blockDim.x + threadIdx.x;
for (int i = index * chunk_size_per_thread; i < (index + 1) * chunk_size_per_thread - 1; i++) {
if (i + 2 * image_width + 1 < image_width * image_height) {
if (i == 0 && blockIdx.x == 0 && blockIdx.x == 0) {
continue;
} else {
y_gradient[i] =
y_kernel[0][0] * channel[i - 1] +
y_kernel[1][0] * channel[i] +
y_kernel[2][0] * channel[i + 1] +
y_kernel[0][1] * channel[i + image_width - 1] +
y_kernel[1][1] * channel[i + image_width] +
y_kernel[2][1] * channel[i + image_width + 1] +
y_kernel[0][2] * channel[i + 2 * image_width - 1] +
y_kernel[1][2] * channel[i + 2 * image_width] +
y_kernel[2][2] * channel[i + 2 * image_width + 1];
}
}
}
return;
}