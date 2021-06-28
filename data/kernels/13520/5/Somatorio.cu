#include "includes.h"
__global__ void Somatorio( float *input, float *results, long int n ) {
extern __shared__ float sdata[];
int idx = blockIdx.x * blockDim.x + threadIdx.x, tx = threadIdx.x;
float x = 0.;
if( idx < n ) {
x = input[ idx ];
}
sdata[ tx ] = x;
__syncthreads( );
for( int offset = blockDim.x / 2; offset > 0; offset >>= 1 ) {
if( tx < offset ) {
sdata[ tx ] += sdata[ tx + offset ];
}
__syncthreads( );
}
if( threadIdx.x == 0 ) {
results[ blockIdx.x ] = sdata[ 0 ];
}
}