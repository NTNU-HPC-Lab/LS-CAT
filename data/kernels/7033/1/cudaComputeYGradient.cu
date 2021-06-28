#include "includes.h"

cudaEvent_t start, stop;




__global__ void cudaComputeYGradient(int* y_gradient, unsigned char* channel, int image_width, int image_height) {
int y_kernel[3][3] = { { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } };
int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index == 0) {
return;
}
y_gradient[index] =
y_kernel[0][0] * channel[index - 1] +
y_kernel[1][0] * channel[index] +
y_kernel[2][0] * channel[index + 1] +
y_kernel[0][1] * channel[index + image_width - 1] +
y_kernel[1][1] * channel[index + image_width] +
y_kernel[2][1] * channel[index + image_width + 1] +
y_kernel[0][2] * channel[index + 2 * image_width - 1] +
y_kernel[1][2] * channel[index + 2 * image_width] +
y_kernel[2][2] * channel[index + 2 * image_width + 1];
return;
}