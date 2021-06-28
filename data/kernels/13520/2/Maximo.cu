#include "includes.h"

#define tam 1.0
#define dx 0.00001
#define dt 0.000001
#define T 0.01
#define kappa 0.000045





__global__ void Maximo( double *input, double *results, int n ) {
extern __shared__ double sdata[];
int idx = blockIdx.x * blockDim.x + threadIdx.x, tx = threadIdx.x;
double x = 0.;
if( idx < n ) {
x = input[ idx ];
}
sdata[ tx ] = x;
__syncthreads( );
for( int offset = blockDim.x / 2; offset > 0; offset >>= 1 ) {
if( tx < offset ) {
if( sdata[ tx ] < sdata[ tx + offset ] ) {
sdata[ tx ] = sdata[ tx + offset ];
}
}
__syncthreads( );
}
if( threadIdx.x == 0 ) {
results[ blockIdx.x ] = sdata[ 0 ];
}
}