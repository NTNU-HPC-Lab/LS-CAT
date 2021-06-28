#include "includes.h"
__global__ void fourier_transform(float *in, float *out, int height, int width, int blockConfig) {
// block elements and function variables
int my_x, k, t;
my_x = blockIdx.x * blockDim.x + threadIdx.x;

// iterate through each element, going from frequency to time domain
for (k = 0; k < height; k++) {
// difference, which will be used to subtract off
float realSum = 0.0;
// iterate through the input element
for (t = 0; t < width; t++) {
// calculate the angle and update the sum
float angle = 2 * M_PI * (my_x * height + t) * (my_x * width + k) / height;
realSum += in[my_x * height + t] * cos(angle);
}
// each output element will be the current sum for that index
out[my_x * height + k] = realSum;
}
}