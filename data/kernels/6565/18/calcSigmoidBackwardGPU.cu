#include "includes.h"
__device__ float activator_derivative( float x )
{
float sig = 1.0f / (1.0f + exp( -x ));
return sig * (1 - sig);
}
__global__ void calcSigmoidBackwardGPU( float *dz_next_layer, float *dz_in, float *dz, float *in, int elements )
{
int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

if( id < elements ){
float x = dz_in[id] += dz_next_layer[id];
float sig = 1.0f / (1.0f + exp( -x ));
dz[id] +=  ( sig * (1 - sig) ) * dz_in[id]; // sigmoid_derivative * dz_in
}

/* original
for( int i = 0; i < dz_in.size.b * dz_in.size.x * dz_in.size.y * dz_in.size.z; ++i ){
dz_in.data[i] += dz_next_layer.data[i];
}

for ( int i = 0; i < in_total_size; ++i ){
dz.data[i] += activator_derivative( in.data[i] ) * dz_in.data[i];
}
*/
}