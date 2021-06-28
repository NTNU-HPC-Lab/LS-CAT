#include "includes.h"
__global__ void unroll_kernel(int h_in, int w_in, int k, float *x, float *x_unroll) {
int w_out_, h_out_, h_unroll, w_unroll_, p, q;
int t = blockIdx.x * 1024 + threadIdx.x; // Index of this thread
int w_out = w_in - k + 1;                // Output image size
int w_unroll = w_out * w_out;            // Unroll limit

if (t < w_unroll) {
h_out_ = t / w_out;                  // Output height
w_out_ = t % w_out;                  // Output width
w_unroll_ = h_out_ * w_out + w_out_; // The index of output pixel in image
for (p = 0; p < k; p++)
for (q = 0; q < k; q++) {
h_unroll = p * k + q;
if ((h_out_ + p) < h_in && (w_out_ + q) < w_in)
x_unroll[h_unroll * w_unroll + w_unroll_] =
x[(h_out_ + p) * w_in + w_out_ + q];
}
}
}