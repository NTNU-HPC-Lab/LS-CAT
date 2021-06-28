#include "includes.h"
__global__ void set_coords_2D(float* coords, size_t y, size_t x){
size_t index = blockIdx.x * blockDim.x + threadIdx.x;
size_t id_x = index % x;
size_t id_y = index / x;
if(index < x * y){
coords[id_x + id_y * x] = id_y - (float)y/2.0;
coords[id_x + id_y * x + x*y] = id_x - (float)x/2.0;
}
__syncthreads();
}