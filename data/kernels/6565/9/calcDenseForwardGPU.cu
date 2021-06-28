#include "includes.h"
__global__ void calcDenseForwardGPU( float *in, float *out, float *weights, float *biases, int batch_size, int in_size_x, int in_size_y, int in_size_z, int out_size_x, int out_size_y, int out_size_z )
{
int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
int id_out = id;
if ( id_out < batch_size * out_size_x * out_size_y * out_size_z ){
int n = id % out_size_x;
id /= out_size_x;
// int y = id % out_size_y;
id /= out_size_y;
// int z = id % out_size_z;
id /= out_size_z;
int b = id;

int w_size_x = in_size_x*in_size_y*in_size_z;

float sum = 0;
for ( int k = 0; k < in_size_z; ++k ){
for ( int j = 0; j < in_size_y; ++j ){
for ( int i = 0; i < in_size_x; ++i ){
int m = k * (in_size_x * in_size_y) + j * (in_size_x) + i;
int w_index = n * (w_size_x) + m;
int in_index = b * (in_size_x * in_size_y * in_size_z) + k * (in_size_x * in_size_y) + j * in_size_x + i;
sum += in[in_index] * weights[w_index];
}
}
}
int bias_index = n;
out[id_out] = sum + biases[bias_index];
}

/* original
for ( int b = 0; b < in.size.b; ++b ){
for ( int n = 0; n < out.size.x; ++n ){
float sum = 0;
for ( int z = 0; z < in.size.z; ++z ){
for ( int j = 0; j < in.size.y; ++j ){
for ( int i = 0; i < in.size.x; ++i ){
int m = map( { 0, i, j, z } );
sum += in( b, i, j, z ) * weights( 0, m, n, 0 );
}
}
}
out( b, n, 0, 0 ) = sum + biases( 0, 0, n, 0);
}
}
*/
}