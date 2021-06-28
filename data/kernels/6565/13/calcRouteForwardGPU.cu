#include "includes.h"
__global__ void calcRouteForwardGPU(float *in, float *out, int in_size_x, int in_size_y, int in_size_z, int z_offset, int elements )
{
// int i = blockIdx.x*blockDim.x + threadIdx.x;
int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

if( id < elements ){
int id_in = id;

int x = id % in_size_x;
id /= in_size_x;
int y = id % in_size_y;
id /= in_size_y;
int z = id % in_size_z;
id /= in_size_z;
int b = id;

int id_out = b * (in_size_z * in_size_x * in_size_y) + (z + z_offset) * (in_size_x * in_size_y) + y * (in_size_x) + x;
out[id_out] = in[id_in];
}

/* original code
for ( int b = 0; b < layer_in.size.b; ++b ){
for ( int z = 0; z < layer_in.size.z; ++z ){
for ( int y = 0; y < layer_in.size.y; y++ ){
for ( int x = 0; x < layer_in.size.x; x++ ){
out( b, x, y, z_offset+z ) = layer_in( b, x, y, z );
}
}
}
}
*/

}