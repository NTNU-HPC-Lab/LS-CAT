#include "includes.h"
__global__ void bcnn_grad_scales_kernel(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates) {
__shared__ float part[BCNN_CUDA_THREADS];
int i, b;
int filter = blockIdx.x;
int p = threadIdx.x;
float sum = 0;
for (b = 0; b < batch; ++b) {
for (i = 0; i < size; i += BCNN_CUDA_THREADS) {
int index = p + i + size * (filter + n * b);
sum += (p + i < size) ? delta[index] * x_norm[index] : 0;
}
}
part[p] = sum;
__syncthreads();
if (p == 0) {
for (i = 0; i < BCNN_CUDA_THREADS; ++i)
scale_updates[filter] += part[i];
}
}