#include "includes.h"
__global__ void init_segmented_rpt(int *d_nnz_num, int *d_seg_rpt, int total_pad_row_num)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i > total_pad_row_num) {
return;
}

if (i == 0) {
d_seg_rpt[i] = 0;
}

else {
d_seg_rpt[i] = d_nnz_num[i - 1];
}
}