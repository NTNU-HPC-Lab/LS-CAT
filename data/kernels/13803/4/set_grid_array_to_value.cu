#include "includes.h"






__global__ void set_grid_array_to_value(float *arr, float value, int N_grid){
int i = blockIdx.x*blockDim.x + threadIdx.x;
int j = blockIdx.y*blockDim.y + threadIdx.y;
int k = blockIdx.z*blockDim.z + threadIdx.z;
int index = k*N_grid*N_grid + j*N_grid + i;

if((i<N_grid) && (j<N_grid) && (k<N_grid)){
arr[index] = value;
}
}