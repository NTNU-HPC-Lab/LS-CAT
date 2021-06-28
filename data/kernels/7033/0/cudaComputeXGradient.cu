#include "includes.h"

cudaEvent_t start, stop;




__global__ void cudaComputeXGradient(int* x_gradient, unsigned char* channel, int image_width, int image_height) {
int x_kernel[3][3] = { { 1, 0, -1 }, { 2, 0, -2 }, { 1, 0, -1 } };
int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index == 0) {
return;
}
x_gradient[index] =
x_kernel[0][0] * channel[index - 1] +
x_kernel[1][0] * channel[index] +
x_kernel[2][0] * channel[index + 1] +
x_kernel[0][1] * channel[index + image_width - 1] +
x_kernel[1][1] * channel[index + image_width] +
x_kernel[2][1] * channel[index + image_width + 1] +
x_kernel[0][2] * channel[index + 2 * image_width - 1] +
x_kernel[1][2] * channel[index + 2 * image_width] +
x_kernel[2][2] * channel[index + 2 * image_width + 1];
return;
}