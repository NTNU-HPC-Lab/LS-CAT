#include "includes.h"
__global__ void max_norm_matrix(float* mat1, int row, int col, int* norm, float* final_norm){
*norm = 0;
__syncthreads();
int id = blockIdx.x * blockDim.x + threadIdx.x;
int size = row*col;
if(id<size){
atomicMax(norm, __float_as_int(abs(mat1[id])));
}
__syncthreads();
if(id==0){
*final_norm = __int_as_float(*norm);
}
}