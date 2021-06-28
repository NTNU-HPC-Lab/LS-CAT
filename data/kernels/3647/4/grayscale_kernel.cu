#include "includes.h"

#define IDX2D(a, i, stride, j) ((a)[(i)*(stride) + (j)])

__global__ void grayscale_kernel(double *z, unsigned char *output, size_t size, double z_min, double z_max) {
const double grid_size = blockDim.x*gridDim.x;
const int idx = threadIdx.x + blockDim.x*blockIdx.x;

for (int i = idx; i < size; i += grid_size)
output[i] = (char) round((z[i]-z_min)/(z_max-z_min)*255);
}