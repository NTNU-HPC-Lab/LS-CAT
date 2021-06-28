#include "includes.h"
__global__ void calcConvolutionForwardGPU( float *out, float *padded_in, float *filters, int padded_in_size_x, int padded_in_size_y, int padded_in_size_z, int batch_size, int out_size_x, int out_size_y, int out_size_z, int kernel_size, int stride, int filter_size)
{
int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
int id_out = id;

if (id_out < batch_size * out_size_x * out_size_y * out_size_z) {
int x = id % out_size_x;
id /= out_size_x;
int y = id % out_size_y;
id /= out_size_y;
int filter = id % out_size_z;
id /= out_size_z;
int b = id;

int mapped_x = x * stride;
int mapped_y = y * stride;

float sum = 0.0;
for ( int z = 0; z < padded_in_size_z; ++z ){ // padded_in_size_z = in_size_z
for ( int j = 0; j < kernel_size; ++j ){
for ( int i = 0; i < kernel_size; ++i ){

int padded_in_index = b * (padded_in_size_x * padded_in_size_y * padded_in_size_z) + z * (padded_in_size_x * padded_in_size_y) + (mapped_y + j) * (padded_in_size_x) + (mapped_x + i);
int filter_index = z * (kernel_size * kernel_size) + j * kernel_size + i;

sum += filters[filter * filter_size + filter_index] * padded_in[padded_in_index];
}
}
}
out[id_out] = sum;
}

/* original code
for ( int b = 0; b < in.size.b; ++b ){
int filters_size = filters.size();
for ( int filter = 0; filter < filters_size; ++filter ){
TensorObject<float> filter_data = filters[filter];
for ( int y = 0; y < out.size.y; ++y ){
for ( int x = 0; x < out.size.x; ++x ){
TensorCoordinate mapped = map_to_input( { 0, (uint16_t)x, (uint16_t)y, 0 }, 0 );
float sum = 0;
for ( int z = 0; z < in.size.z; ++z ){
for ( int j = 0; j < kernel_size; ++j ){
for ( int i = 0; i < kernel_size; ++i ){
sum += filter_data( 0, i, j, z ) * padded_in( b, mapped.x + i, mapped.y + j, z );
}
}
}
out( b, x, y, filter ) = sum;
}
}
}
}*/
}