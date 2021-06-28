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
__global__ void needle_cuda_noshr_2( int* reference, int* matrix_cuda, int cols, int penalty, int i, int block_width)
{

int bx = blockIdx.x;
int tx = threadIdx.x;

int b_index_x = bx + block_width - i;
int b_index_y = block_width - bx -1;

int index    = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + ( cols + 1 );

for( int m = 0 ; m < BLOCK_SIZE ; m++) {
if ( tx <= m ){
int t_index_x = tx;
int t_index_y = m - tx;
int idx = index + t_index_y * cols + t_index_x;
matrix_cuda[idx] = maximum( matrix_cuda[idx-cols-1] + reference[idx],
matrix_cuda[idx - 1]    - penalty,
matrix_cuda[idx - cols] - penalty);
}
}

for( int m = BLOCK_SIZE - 2 ; m >=0 ; m--) {
if ( tx <= m){
int t_index_x =  tx + BLOCK_SIZE - m -1;
int t_index_y =  BLOCK_SIZE - tx - 1;
int idx = index + t_index_y * cols + t_index_x;
matrix_cuda[idx] = maximum( matrix_cuda[idx-cols-1] + reference[idx],
matrix_cuda[idx - 1]    - penalty,
matrix_cuda[idx - cols] - penalty);
}
}
}