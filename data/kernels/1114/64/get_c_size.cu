#include "includes.h"
__global__ void get_c_size(int *d_c_size, int *d_full_cl, int size)
{
int i = blockDim.x * blockIdx.x + threadIdx.x;
if (i >= size) {
return;
}

if (d_full_cl[i] != 0) {
atomicAdd(d_c_size, 1);
}
}