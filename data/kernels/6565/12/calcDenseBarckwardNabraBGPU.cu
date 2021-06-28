#include "includes.h"
__global__ void calcDenseBarckwardNabraBGPU( float *dz_in, float *dB, int batch_size, int out_size_x ){
int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

if( id < out_size_x ){
for( int b = 0; b < batch_size; ++b ){
dB[id] += dz_in[ b * (out_size_x) + id ];
}
}
/* original
for ( int n = 0; n < out.size.x; ++n ){
for( int b = 0; b < in.size.b; ++b ){
dB( 0, 0, n, 0 ) += dz_in( b, n, 0, 0 );
}
}
*/
}