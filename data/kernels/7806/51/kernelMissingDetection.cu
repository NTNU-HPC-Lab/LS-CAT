#include "includes.h"
__global__ void kernelMissingDetection(int nVerts, int *cactive, int *cvertarr) {
int x = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

// Check for missing sites
if (x < nVerts && cvertarr[x] < 0)
cactive[x] = 0;
}