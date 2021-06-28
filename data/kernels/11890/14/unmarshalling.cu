#include "includes.h"
__global__ void unmarshalling(int *input_itemsets, int *tmp, int max_rows, int max_cols)
{
int i, j;

i = blockIdx.y*blockDim.y+threadIdx.y;
j = blockIdx.x*blockDim.x+threadIdx.x;

if( i >= max_rows || j >= max_cols) return;

if( (i+j) < max_rows) {
input_itemsets[i*max_cols+j] = tmp[(i+j)*max_cols+j];
}
else {
input_itemsets[i*max_cols+j] = tmp[(i+j)*max_cols+j-(i+j-max_rows+1)];
}

}