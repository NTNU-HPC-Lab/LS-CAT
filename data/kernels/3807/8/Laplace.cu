#include "includes.h"
__global__ void Laplace(float* d_out, float* d_in) {
int rowID = blockIdx.x + 1;
int colID = threadIdx.x + 1;
int pos = rowID * (blockDim.x + 2) + colID;
d_out[pos] = (d_in[pos - 1] + d_in[pos + 1] +
d_in[pos - blockDim.x - 2] + d_in[pos + blockDim.x + 2]) /  4.;
}