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
__global__ void lower_right(int *input_itemsets, int *reference, int *tmp, int max_rows, int max_cols, int i, int penalty)
{

int r, c;

r = i;
c = blockIdx.x*blockDim.x+threadIdx.x;

if( c >= (max_cols-(i-max_rows+1))) return;

tmp[r*max_cols+c] = maximum( tmp[(r-2)*max_cols+c+1]+ reference[(max_rows-1-c)*max_cols+c+(i-max_rows+1)],
tmp[(r-1)*max_cols+c] - penalty,
tmp[(r-1)*max_cols+c+1] - penalty);
}