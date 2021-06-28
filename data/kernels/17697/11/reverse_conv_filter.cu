#include "includes.h"
extern "C" {
}
__global__ void reverse_conv_filter(const float* x, float beta, float* y, unsigned int filter_len, unsigned int len) {
int tid = blockIdx.x*blockDim.x + threadIdx.x;
if (tid < len) {
if (beta == 0.0f) {
for(int i = 0; i < filter_len; ++i) {
y[tid*filter_len + i] = x[tid*filter_len + ((filter_len-1) - i)];
}
}
else {
for(int i = 0; i < filter_len; ++i) {
y[tid*filter_len + i] = x[tid*filter_len + ((filter_len-1) - i)] + beta * y[tid*filter_len + i];
}
}
}
}