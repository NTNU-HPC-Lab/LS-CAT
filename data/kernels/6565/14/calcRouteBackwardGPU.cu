#include "includes.h"
__global__ void calcRouteBackwardGPU( float *dz_in, float *dz, int in_size_x, int in_size_y, int in_size_z, int z_offset, int elements )
{
int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

if( id < elements ){
int id_out = id;
int x = id % in_size_x;
id /= in_size_x;
int y = id % in_size_y;
id /= in_size_y;
int z = id % in_size_z;
id /= in_size_z;
int b = id;

int id_in = b * (in_size_z * in_size_x * in_size_y) + (z + z_offset) * (in_size_x * in_size_y) + y * (in_size_x) + x;
dz[id_out] += dz_in[id_in];
}
/*
for ( int b = 0; b < layer_dz.size.b; ++b ){
for ( int z = 0; z < layer_dz.size.z; ++z ){
for ( int y = 0; y < layer_dz.size.y; ++y ){
for ( int x = 0; x < layer_dz.size.x; ++x ){
layer_dz( b, x, y, z ) += dz_in( b, x, y, z_offset+z );
}
}
}
}
*/
}