#include "includes.h"
__global__ void marshalling1(int *input_itemsets, int *tmp, int max_rows, int max_cols)
{
int i, j;

i = blockIdx.y*blockDim.y+threadIdx.y;
j = blockIdx.x*blockDim.x+threadIdx.x;

if( i >= max_rows || j >= max_cols) return;

if( j <= i) {
tmp[i*max_cols+j] = input_itemsets[(i-j)*max_cols+j];
}
else {
tmp[i*max_cols+j] = 0;
}
}