#include "includes.h"
__global__ void set_cl(int *nnz_num, int *cl, int chunk, int pad_M)
{
int c_size = pad_M / chunk;
int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i >= c_size) {
return;
}
int offset = chunk * i;
int max = 0;
int j, length;
for (j = 0; j < chunk; j++) {
length = nnz_num[offset + j];
if (length > max) {
max = length;
}
}
cl[i] = max;
}