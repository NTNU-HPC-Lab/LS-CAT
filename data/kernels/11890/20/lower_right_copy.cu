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
__global__ void lower_right_copy(int *dst, int *input_itemsets, int *reference, int max_rows, int max_cols, int i, int penalty)
{
int r, c;

r = blockIdx.y*blockDim.y+threadIdx.y+i+1;
c = blockIdx.x*blockDim.x+threadIdx.x+i+1;

if( r >= max_rows || c >= max_cols) return;

if( r == (max_cols - c + i)) {
dst[r*max_cols+c]
= maximum( input_itemsets[(r-1)*max_cols+c-1]+ reference[r*max_cols+c],
input_itemsets[r*max_cols+c-1] - penalty,
input_itemsets[(r-1)*max_cols+c] - penalty);
}
else {
dst[r*max_cols+c] = input_itemsets[r*max_cols+c];
}
}