#include "includes.h"
__global__ void calcConvolutionBackwardResetGradGPU( float *filter_grads, int in_size_z, int kernel_size, int filter_size, int elements )
{
int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

if ( id < elements ) {
int i = id % kernel_size;
id /= kernel_size;
int j = id % kernel_size;
id /= kernel_size;
int z = id % in_size_z;
id /= in_size_z;
int filter = id;

int filter_grad_index = (filter * (in_size_z * kernel_size * kernel_size) + z * (kernel_size * kernel_size) + j * kernel_size + i) * 2;
filter_grads[ filter_grad_index ] = 0;
}

/* original code
int k_end = filter_grads.size();
int kernel_size_2 = kernel_size * kernel_size;
int i_end = kernel_size_2 * in.size.z;
for ( int k = 0; k < k_end; ++k ){
for ( int i = 0; i < i_end ; ++i ){
filter_grads[k].data[i].grad = 0;
}
}
*/
}