#include "includes.h"
__global__ void set_d_check_nnz(int *d_check_nnz, int *d_nnz_num, int pad_M, int SIGMA, int sigma_block_row)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i >= pad_M) {
return;
}

int a = 1;
if (d_nnz_num[blockIdx.y * pad_M + i] > 0) {
atomicAdd(&(d_check_nnz[blockIdx.y * sigma_block_row + i / SIGMA]), a);
}
}