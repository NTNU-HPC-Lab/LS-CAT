#include "includes.h"
__global__ void inverse_transform(float *in, float *out, int height, int width) {
// block elements
int my_x, k, t;
my_x = blockIdx.x * blockDim.x + threadIdx.x;

// iterate through each element, going from frequency to time domain
for (k = 0; k < height; k++) {
// difference, which will be used to subtract off
float realSum = 0;
// iterate through the input element
for (t = 0; t < width; t++) {
float angle = 2 * M_PI * (my_x * height + t) * (my_x * height + k) / height;
realSum += in[my_x * height + t] * cos(angle);
}
out[my_x * height + k] = (realSum / height);
}
}