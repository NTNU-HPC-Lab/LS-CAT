#include "includes.h"
__global__ void recenter_2D(float* coords, size_t dim_y, size_t dim_x){
size_t index = blockIdx.x * blockDim.x + threadIdx.x;
if(index < dim_x * dim_y){
coords[index] += (float)dim_y/2.0;
coords[index + dim_x*dim_y] += (float)dim_x/2.0;
}
__syncthreads();
}