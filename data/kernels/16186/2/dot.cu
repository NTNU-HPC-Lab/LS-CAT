#include "includes.h"
__global__ void dot( int *a, int *b, int *c ) {
__shared__ int temp[THREADS_PER_BLOCK];
int index = threadIdx.x + blockIdx.x * blockDim.x;
temp[threadIdx.x] = a[index] * b[index];
__syncthreads();
if( 0 == threadIdx.x ) {
int sum = 0;
for( int i = 0; i < THREADS_PER_BLOCK; i++ )
sum += temp[i];
atomicAdd( c , sum );
}
}