#include "includes.h"
__global__ void calcSoftmaxBackwardGPU( float *dz_next_layer, float *dz_in, float *dz, unsigned int n )
{
int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
// unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

if ( index < n ){
dz_in[index] += dz_next_layer[index];
dz[index] +=  dz_in[index];
}

/* original
for( int i = 0; i < dz_in.size.b * dz_in.size.x * dz_in.size.y * dz_in.size.z; ++i ){
dz_in.data[i] += dz_next_layer.data[i];
}

for ( int i = 0; i < in.size.b * in.size.x * in.size.y * in.size.z; ++i ){
dz.data[i] += dz_in.data[i];
}
*/
}