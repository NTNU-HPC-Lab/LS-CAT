#include "includes.h"

__global__ void binZeros(int *d_bin_count, int bin_size){
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < bin_size){
d_bin_count[i] = 0;
}
}