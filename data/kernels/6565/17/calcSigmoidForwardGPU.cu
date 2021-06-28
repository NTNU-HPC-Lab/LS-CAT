#include "includes.h"
__global__ void calcSigmoidForwardGPU(float *in, float *out, int elements)
{
int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if( id < elements ){
float v = in[id];
v = 1.0f / (1.0f + exp( -v )); // sigmoid
out[id] = v;
}

/* original
for ( int i = 0; i < in_total_size; ++i ){
out.data[i] = activator_function(in.data[i]);
}
*/
}