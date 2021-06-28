#include "includes.h"
__global__ void calcSoftmaxDivForwardGPU(float *out, float *sum, int batch_size, int in_size_x, unsigned int n)
{
// int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
if(index<n && *(sum + blockIdx.x)>0.0){
// out[id] = out[id] / *sum;
out[index] = out[index] / *(sum + blockIdx.x);
}

/* original
for ( int i = 0; i < in.size.x; ++i ){
out( b, i, 0, 0 ) = out( b, i, 0, 0 ) / sum;
}
*/
}