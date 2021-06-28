#include "includes.h"
__global__ void marshalling2(int *input_itemsets, int *tmp, int max_rows, int max_cols)
{
int i, j;

i = blockIdx.y*blockDim.y+threadIdx.y+max_rows;
j = blockIdx.x*blockDim.x+threadIdx.x;

if( i >= max_rows*2-1 || j >= max_cols) return;

if( j < max_cols-(i-max_rows+1)) {
tmp[i*max_cols+j] = input_itemsets[(max_rows-1-j)*max_cols+j+1+(i-max_rows)];
}
else {
tmp[i*max_cols+j] = 0;
}
}