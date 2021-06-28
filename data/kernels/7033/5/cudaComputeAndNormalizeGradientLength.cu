#include "includes.h"
__global__ void cudaComputeAndNormalizeGradientLength(unsigned char *channel_values, int* x_gradient, int* y_gradient, int chunk_size_per_thread) {
int index = blockIdx.x * blockDim.x + threadIdx.x;
for (int i = index * chunk_size_per_thread; i < (index + 1) * chunk_size_per_thread; i++) {
int gradient_length = int(sqrt(float(x_gradient[i] * x_gradient[i] + y_gradient[i] * y_gradient[i])));
if (gradient_length > 255) {
gradient_length = 255;
}
channel_values[i] = gradient_length;
}
return;
}