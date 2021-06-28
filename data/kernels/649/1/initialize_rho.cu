#include "includes.h"
__global__ void initialize_rho(float* rho, int size_c, int nc) {
int i = blockIdx.x*blockDim.x + threadIdx.x;
int c = blockIdx.y*blockDim.y + threadIdx.y;
if (i < size_c && c < nc) {
rho[c*(size_c)+i] = 0.5f;
}
}