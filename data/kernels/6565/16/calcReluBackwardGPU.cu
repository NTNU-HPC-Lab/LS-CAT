#include "includes.h"
__global__ void calcReluBackwardGPU( float *dz_next_layer, float *dz_in, float *dz, float *in, int elements )
{
int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if( id < elements ){
dz_in[id] += dz_next_layer[id];
dz[id] += (in[id] < 0) ? (0) : (1.0 * dz_in[id]);
}

/* original
for( unsigned i = 0; i < data_size; ++i ){
dz_in.data[i] += dz_next_layer.data[i];
dz.data[i] +=  (in.data[i] < 0) ? (0) : (1.0 * dz_in.data[i]);
}
*/
}