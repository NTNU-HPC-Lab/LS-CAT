#include "includes.h"
__global__ void downSampleKernel(unsigned char * d_in, unsigned char * d_out, size_t skip) {
size_t i = threadIdx.x;
// Assuming 3 channels BGR and averaging
int px = d_in[i * skip * 3] + d_in[i * skip * 3 + 1] + d_in[i * skip * 3 + 2];
d_out[i] = px / 3;
}