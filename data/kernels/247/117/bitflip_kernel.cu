#include "includes.h"
__global__ void bitflip_kernel(float* M, int height, int row, int n) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
int off = blockDim.x * gridDim.x;
for (unsigned int i = idx; i < n; i += off){
M[i * height + row] = 1 - M[i * height + row];
}

}