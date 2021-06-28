#include "includes.h"
__global__ void set_bin(int *d_row_nz, int *d_bin_size, int *d_max, int M, int min, int mmin)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i >= M) {
return;
}
int nz_per_row = d_row_nz[i];

atomicMax(d_max, nz_per_row);

int j = 0;
for (j = 0; j < BIN_NUM - 2; j++) {
if (nz_per_row <= (min << j)) {
if (nz_per_row <= (mmin)) {
atomicAdd(d_bin_size + j, 1);
}
else {
atomicAdd(d_bin_size + j + 1, 1);
}
return;
}
}
atomicAdd(d_bin_size + BIN_NUM - 1, 1);
}