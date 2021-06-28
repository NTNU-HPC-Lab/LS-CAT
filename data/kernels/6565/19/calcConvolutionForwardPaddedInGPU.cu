#include "includes.h"
__global__ void calcConvolutionForwardPaddedInGPU( float *in, float *padded_in, int batch_size, int in_size_x, int in_size_y, int in_size_z, int padding)
{
int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

if( id < batch_size * in_size_x * in_size_y * in_size_z ){
int in_index = id;

int x = id % in_size_x;
id /= in_size_x;
int y = id % in_size_y;
id /= in_size_y;
int z = id % in_size_z;
id /= in_size_z;
int b = id;

int pad_index = b * (in_size_z * (in_size_x + 2*padding) * (in_size_y + 2*padding) ) +
z * ((in_size_x + 2*padding) * (in_size_y + 2*padding)) +
(y+padding) * (in_size_x + 2*padding) +
(x+padding) ;

padded_in[pad_index] = in[in_index];
}
/* original code
for ( int b = 0; b < in.size.b; ++b ){
for ( int z = 0; z < in.size.z; ++z ){
for ( int y = 0; y < in.size.y; ++y ){
for ( int x = 0; x < in.size.x; ++x ){
padded_in( b, padding+x, padding+y, z ) = in( b, x, y, z );
}
}
}
}
*/
}