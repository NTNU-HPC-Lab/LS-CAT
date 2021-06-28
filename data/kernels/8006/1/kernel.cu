#include "includes.h"
__global__ void kernel( int *a, int dimx, int dimy ) {
int i = blockIdx.x*blockDim.x + threadIdx.x;
a[i] = blockIdx.x * dimx + threadIdx.x;
}