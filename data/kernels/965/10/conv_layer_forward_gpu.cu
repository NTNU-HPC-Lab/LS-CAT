#include "includes.h"
__global__ void conv_layer_forward_gpu(float *x, float *w, float *y, int h_in, int w_in, int w_out, int k, int m) {
int n, m_, h, w_, p, q;
n = blockIdx.x;   // Batch index
m_ = blockIdx.y;  // Channel index
h = threadIdx.y;  // Pixel (h, w_)
w_ = threadIdx.x; // Pixel (h, w_)
float ans = 0;    // Return value
int offset = n * (h_in * w_in);

// Load w into shared memory to speed up the data access
__shared__ float cached_w[CONV_KERNEL_SIZE][CONV_KERNEL_SIZE];
if (h < k && w_ < k) {
cached_w[h][w_] = w[m_ * (k * k) + h * k + w_];
}
__syncthreads();

// Loop over k by k kernel
if (h < w_out && w_ < w_out) {
for (p = 0; p < k; p++) {
for (q = 0; q < k; q++)
ans = ans + x[offset + (h + p) * w_in + (w_ + q)] * cached_w[p][q];
}
// Write out the return value
y[n * (m * w_out * w_out) + m_ * (w_out * w_out) + h * w_out + w_] = ans;
}
}