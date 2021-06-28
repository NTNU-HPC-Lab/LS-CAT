#include "includes.h"
__global__ void value_add_matrix(float* mat1, float* mat2, int row, int col, float v){
int id = blockIdx.x * blockDim.x + threadIdx.x;
int size = row*col;
if(id<size){
mat2[id] = mat1[id] + v;
}
}