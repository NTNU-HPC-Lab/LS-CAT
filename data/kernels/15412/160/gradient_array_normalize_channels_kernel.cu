#include "includes.h"
__global__ void gradient_array_normalize_channels_kernel(float *x, int size, int batch, int channels, int wh_step, float *delta_gpu)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;

int wh_i = i % wh_step;
int b = i / wh_step;

if (i < size) {
int k;
/*
float grad = 0;
for (k = 0; k < channels; ++k) {
const int index = wh_i + k * wh_step + b*wh_step*channels;
float out = x[index];
float delta = delta_gpu[index];
grad += out*fabs(delta);
}
*/
for (k = 0; k < channels; ++k) {
const int index = wh_i + k * wh_step + b*wh_step*channels;
if (x[index] > 0) {
float delta = delta_gpu[index];
float grad = x[index];
delta = delta * grad;
delta_gpu[index] = delta;
}
}
}
}