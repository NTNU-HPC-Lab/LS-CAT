#include "includes.h"
__global__ void matrix_add_matrix(float* mat1, float* mat2, float* mat3, int row, int col, int sign){
int id = blockIdx.x * blockDim.x + threadIdx.x;
int size = row*col;
if(id<size){
mat3[id] = mat1[id] + sign*mat2[id];
}
}