#include "includes.h"
__global__ void calcMaxPoolForwardGPU( float *in,float *out, int in_size_x, int in_size_y, int in_size_z, int batch_size, int out_size_x, int out_size_y, int out_size_z, int stride, int kernel_size )
{
int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
int id_out = id;

if( id_out < batch_size * out_size_x * out_size_y * out_size_z) {
int x = id % out_size_x;
id /= out_size_x;
int y = id % out_size_y;
id /= out_size_y;
int z = id % out_size_z;
id /= out_size_z;
int b = id;

int mapped_x = x * stride;
int mapped_y = y * stride;

float mval = -1000000.0;
for ( int j = 0; j < kernel_size; ++j ){
for ( int i = 0; i < kernel_size; ++i ){

int id_in = b * (in_size_z * in_size_x * in_size_y) +
z * (in_size_x * in_size_y) +
(mapped_y + j) * (in_size_x) +
(mapped_x + i);

float v = in[id_in];
if ( v > mval ){
mval = v;
}
}
}
out[id_out] = mval;
}

/* original
for ( int b = 0; b < in.size.b; ++b ){
for ( int z = 0; z < out.size.z; ++z ){
for ( int y = 0; y < out.size.y; ++y ){
for ( int x = 0; x < out.size.x; ++x ){
TensorCoordinate mapped = map_to_input( { 0, (uint16_t)x, (uint16_t)y, 0 }, 0 );
float mval = -FLT_MAX;
for ( int j = 0; j < kernel_size; ++j ){
for ( int i = 0; i < kernel_size; ++i ){
float v = in( b, mapped.x + i, mapped.y + j, z );
if ( v > mval ){
mval = v;
}
}
}
out( b, x, y, z ) = mval;
}
}
}
}

*/
}