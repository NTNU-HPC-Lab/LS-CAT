#include "includes.h"
__global__ void calcDenseBackwardGPU( float *dz_in, float *dz, float *in, float *weights, float *biases, float *gradients, float *dW, float *dB, int batch_size, int in_size_x, int in_size_y, int in_size_z, int out_size_x, int out_size_y, int out_size_z, float momentum, float decay )
{
int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
int id_out = id;
int id_in  = id / out_size_x;

if( id_out < batch_size * in_size_x * in_size_y * in_size_z * out_size_x ){

int n = id % out_size_x;
id /= out_size_x;
int i = id % in_size_x;
id /= in_size_x;
int j = id % in_size_y;
id /= in_size_y;
int z = id % in_size_z;
id /= in_size_z;
int b = id;

int w_size_x = in_size_x * in_size_y * in_size_z;
// int w_size_y = out_size_x;

int m = z * (in_size_x * in_size_y) + j * (in_size_x) + i;

//    for ( int n = 0; n < out_size_x; ++n ){
float dzin = dz_in[b * (out_size_x * out_size_y * out_size_z) + n];

int w_index = n * w_size_x + m;

float w = weights[w_index];
gradients[ (n*batch_size + b) * 2 ] = dzin;
dz[id_in] += dzin * w;

dW[w_index] += in[id_in] * (gradients[ (n*batch_size + b) * 2 ] + gradients[ (n*batch_size + b) * 2 + 1 ] * momentum) + (decay * w);
//    }

}

/* original
for ( int n = 0; n < out.size.x; ++n ){
for ( int z = 0; z < in.size.z; ++z ){
for ( int j = 0; j < in.size.y; ++j ){
for ( int i = 0; i < in.size.x; ++i ){
int m = map( { 0, i, j, z } );

for( int b = 0; b < in.size.b; ++b ){
GradientObject& grad = gradients[ n*in.size.b + b ];
float dzin = dz_in( b, n, 0, 0 );
float w = weights(0, m, n, 0);
grad.grad = dzin;

dz( b, i, j, z ) += dzin * w;
dW( 0, m, n, 0 ) += in( b, i, j, z ) * (grad.grad + grad.grad_prev * _momentum) + (_decay * w);
}
}
}
}
}
// original to here
*/
}