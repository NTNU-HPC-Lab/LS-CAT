#include "includes.h"

cudaEvent_t start, stop;




__global__ void cudaComputeAndNormalizeGradientLength(unsigned char *channel_values, int* x_gradient, int* y_gradient) {
int index = blockIdx.x * blockDim.x + threadIdx.x;
int gradient_length = int(sqrt(float(x_gradient[index] * x_gradient[index] + y_gradient[index] * y_gradient[index])));
if (gradient_length > 255) {
gradient_length = 255;
}
channel_values[index] = gradient_length;
return;
}