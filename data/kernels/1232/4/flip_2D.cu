#include "includes.h"
__device__ void exchange(float &a, float &b){
float temp = a;
a = b;
b = temp;
}
__global__ void flip_2D(float* coords, size_t dim_y, size_t dim_x, int do_y, int do_x){
size_t index = blockIdx.x * blockDim.x + threadIdx.x;
size_t total = dim_x * dim_y;
size_t id_x = index % dim_x;
size_t id_y = index / dim_x;
if(index < total){
if(do_x && id_x < (dim_x / 2)){
exchange(coords[total + id_y * dim_x + id_x],
coords[total + id_y * dim_x + dim_x-1 - id_x]);
__syncthreads();
}
if(do_y && id_y < (dim_y / 2)){
exchange(coords[id_y * dim_x + id_x], coords[(dim_y-1 - id_y) * dim_x + id_x]);
__syncthreads();
}
}
}