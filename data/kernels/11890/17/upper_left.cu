#include "includes.h"
__device__ __host__ int maximum( int a, int b, int c){
int k;
if( a <= b )
k = b;
else
k = a;

if( k <=c )
return(c);
else
return(k);
}
__global__ void upper_left(int *dst, int *input_itemsets, int *reference, int max_rows, int max_cols, int i, int penalty)
{
int r, c;

r = blockIdx.y*blockDim.y+threadIdx.y+1;
c = blockIdx.x*blockDim.x+threadIdx.x+1;

if( r >= i+1 || c >= i+1) return;

if( r == (i - c + 1)) {
int base = r*max_cols+c;
dst[base] = maximum( input_itemsets[base-max_cols-1]+ reference[base],
input_itemsets[base-1] - penalty,
input_itemsets[base-max_cols] - penalty);
}
}