#include "includes.h"
__global__ void activate_array_normalize_channels_kernel(float *x, int size, int batch, int channels, int wh_step, float *output_gpu)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;

int wh_i = i % wh_step;
int b = i / wh_step;

const float eps = 0.0001;
if (i < size) {
float sum = eps;
int k;
for (k = 0; k < channels; ++k) {
float val = x[wh_i + k * wh_step + b*wh_step*channels];
if (val > 0) sum += val;
}
for (k = 0; k < channels; ++k) {
float val = x[wh_i + k * wh_step + b*wh_step*channels];
if (val > 0) val = val / sum;
else val = 0;
output_gpu[wh_i + k * wh_step + b*wh_step*channels] = val;
}
}
}