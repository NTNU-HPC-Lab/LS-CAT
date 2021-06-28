#include "includes.h"
__global__ void copy(int *dst, int *input_itemsets, int max_rows, int max_cols, int lb0, int lb1, int ub0, int ub1)
{
int r, c;

r = blockIdx.y*blockDim.y+threadIdx.y+lb0;
c = blockIdx.x*blockDim.x+threadIdx.x+lb1;

if( r >= ub0 || c >= ub1) return;

int idx = r*max_cols+c;
dst[idx] = input_itemsets[idx];
}