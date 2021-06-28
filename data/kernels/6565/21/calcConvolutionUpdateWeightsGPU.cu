#include "includes.h"
__global__ void calcConvolutionUpdateWeightsGPU( float *filters, float *filter_grads, int in_size_z, int number_filters, int kernel_size, float momentum, float decay, float learning_rate, int elements )
{
int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

if ( id < elements ) {

int id_out = id;
int i = id % kernel_size;
id /= kernel_size;
int j = id % kernel_size;
id /= kernel_size;
int z = id % in_size_z;
id /= in_size_z;
int filter = id;

int filter_size = 1 * kernel_size * kernel_size * in_size_z;
int filter_grad_index = (filter * filter_size + z * (kernel_size * kernel_size) + j * kernel_size + i) * 2;

float grad = filter_grads[ filter_grad_index ];
float grad_prev = filter_grads[ filter_grad_index + 1 ];
float m = ( grad + grad_prev * momentum );

filter_grads[ filter_grad_index + 1 ] = m;

float w = filters[ id_out ];
w -= learning_rate * ( m + (decay * w));
filters[ id_out ] = w;
}

/* original code
int filters_size = filters.size();
for ( int a = 0; a < filters_size; ++a ){
for ( int z = 0; z < in.size.z; ++z ){
for ( int j = 0; j < kernel_size; ++j ){
for ( int i = 0; i < kernel_size; ++i ){
GradientObject& grad = filter_grads[a].get( 0, i, j, z );
float m = (grad.grad + grad.grad_prev * momentum);
grad.grad_prev = m;
float& w = filters[a].get( 0, i, j, z );
w -= lr * ( m + (decay * w));
}
}
}
}
*/
}