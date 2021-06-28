#include "includes.h"
__global__ void matrixTranspose(unsigned int* A_d, unsigned int *T_d, int rowCount, int colCount) {

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// **** Populate vecADD kernel function ****
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
if (row < rowCount && col < colCount){
T_d[col*rowCount+row] = A_d[row*colCount+col];
}

}