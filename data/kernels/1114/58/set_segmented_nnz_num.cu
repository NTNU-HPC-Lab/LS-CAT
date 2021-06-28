#include "includes.h"
__global__ void set_segmented_nnz_num(int *d_rpt, int *d_col, int *d_nnz_num, int *d_group_seg, int *d_offset, size_t seg_size, size_t seg_num, int M, int pad_M, int group_num_col)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i >= M) {
return;
}

int width = d_rpt[i + 1] - d_rpt[i];

int g, j;
int col;

int offset = d_rpt[i];
int index;

for (j = 0; j < width; j++) {
index = offset + j;
col = d_col[index];
g = col / seg_size;
d_offset[index] = d_nnz_num[g * pad_M + i];
d_nnz_num[g * pad_M + i]++;
d_group_seg[index] = g;
}
}