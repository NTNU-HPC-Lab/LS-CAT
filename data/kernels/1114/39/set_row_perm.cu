#include "includes.h"
__global__ void set_row_perm(int *d_bin_size, int *d_bin_offset, int *d_max_row_nz, int *d_row_perm, int M, int min, int mmin)
{
int i = blockDim.x * blockIdx.x + threadIdx.x;
if (i >= M) {
return;
}

int nz_per_row = d_max_row_nz[i];
int dest;

int j = 0;
for (j = 0; j < BIN_NUM - 2; j++) {
if (nz_per_row <= (min << j)) {
if (nz_per_row <= mmin) {
dest = atomicAdd(d_bin_size + j, 1);
d_row_perm[d_bin_offset[j] + dest] = i;
}
else {
dest = atomicAdd(d_bin_size + j + 1, 1);
d_row_perm[d_bin_offset[j + 1] + dest] = i;
}
return;
}
}
dest = atomicAdd(d_bin_size + BIN_NUM - 1, 1);
d_row_perm[d_bin_offset[BIN_NUM - 1] + dest] = i;

}