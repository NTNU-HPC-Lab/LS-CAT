#include "includes.h"
__global__ void calcReluForwardGPU(float *in, float *out, int elements)
{
int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if( id < elements ){
float v = in[id];
if ( v < 0 ){
v = 0.0;
}
out[id] = v;
}

/* original
for( unsigned i = 0; i < data_size; ++i ){
float v = in.data[i];
if ( v < 0 ){
v = 0;
}
out.data[i] = v;
}
*/
}