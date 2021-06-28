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
__global__ void upper_left(int *input_itemsets, int *reference, int max_rows, int max_cols, int i, int penalty)
{
int idx, r, c;

idx = blockIdx.x*blockDim.x+threadIdx.x;

if( idx >= i) return;

r = i - idx;
c = i + 1 - r;

int base = r*max_cols+c;
input_itemsets[base]
= maximum( input_itemsets[base-max_cols-1]+ reference[base],
input_itemsets[base-1] - penalty,
input_itemsets[base-max_cols] - penalty);
}