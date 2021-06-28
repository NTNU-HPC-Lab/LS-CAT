#include "includes.h"
__global__ void rotate_2D(float* coords, size_t dim_y, size_t dim_x, float cos_angle, float sin_angle){
size_t index = blockIdx.x * blockDim.x + threadIdx.x;
size_t total = dim_x * dim_y;
float new_y, new_x;
float old_y = coords[index];
float old_x = coords[index + total];
if(index < total){
new_y = cos_angle * old_y + sin_angle * old_x;
new_x = -sin_angle * old_y + cos_angle * old_x;
__syncthreads();
coords[index] = new_y;
coords[index + total] = new_x;
__syncthreads();
}
}