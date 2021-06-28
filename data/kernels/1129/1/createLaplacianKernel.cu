#include "includes.h"
__global__ void createLaplacianKernel(float *grid, float *kernel, int nrDimensions, int nrGridElements) {
size_t x = threadIdx.x + blockDim.x * blockIdx.x;

if (x >= nrGridElements)
return;

for(int d = 0; d < nrDimensions; ++d) {
if (d == 0)
kernel[x] = grid[x];
else
kernel[x] += grid[x + d*nrGridElements];
}
}