#include "includes.h"
__global__ void set_coords_3D(float* coords, size_t z, size_t y, size_t x){
size_t index = blockIdx.x * blockDim.x + threadIdx.x;
size_t id_x = index % x;
size_t id_y = (index / x) % y;
size_t id_z = index / (x * y);
if(index < x * y * z){
coords[index] = id_z - (float)z/2.0;
coords[index + x * y * z] = id_y - (float)y/2.0;
coords[index + 2 * x * y * z] = id_x -(float)x/2.0;
}
__syncthreads();
}