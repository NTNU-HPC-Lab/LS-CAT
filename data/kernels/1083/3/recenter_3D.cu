#include "includes.h"
__global__ void recenter_3D(float* coords, size_t dim_z, size_t dim_y, size_t dim_x){
size_t index = blockIdx.x * blockDim.x + threadIdx.x;
size_t total = dim_x * dim_y * dim_z;
if(index < total){
coords[index] += (float)dim_z/2.0;
coords[index + total] += (float)dim_y/2.0;
coords[index + 2 * total] += (float)dim_x/2.0;
}
__syncthreads();
}