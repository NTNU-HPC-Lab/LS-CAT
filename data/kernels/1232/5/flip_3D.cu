#include "includes.h"
__device__ void exchange(float &a, float &b){
float temp = a;
a = b;
b = temp;
}
__global__ void flip_3D(float* coords, size_t dim_z, size_t dim_y, size_t dim_x, int do_z, int do_y, int do_x){
size_t index = blockIdx.x * blockDim.x + threadIdx.x;
size_t total = dim_x * dim_y * dim_z;
size_t total_xy = dim_x * dim_y;
size_t id_x = index % dim_x;
size_t id_y = (index / dim_x) % dim_x;
size_t id_z = index / (dim_x * dim_y);
if(index < total){
if(do_x && id_x < (dim_x / 2)){
exchange(coords[2 * total + id_z * total_xy + id_y * dim_x + id_x],
coords[2 * total + id_z * total_xy + id_y * dim_x + dim_x-1 - id_x]);
__syncthreads();
}
if(do_y && id_y < (dim_y / 2)){
exchange(coords[total + id_z * total_xy + id_y * dim_x + id_x],
coords[total + id_z * total_xy + (dim_y-1 - id_y) * dim_x + id_x]);
__syncthreads();
}
if(do_z && id_z < (dim_z / 2)){
exchange(coords[id_z * total_xy + id_y * dim_x + id_x],
coords[(dim_z-1 -id_z) * total_xy + id_y * dim_x + id_x]);
__syncthreads();
}
}
}