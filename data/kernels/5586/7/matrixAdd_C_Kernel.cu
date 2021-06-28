#include "includes.h"
__global__ void matrixAdd_C_Kernel(float* A, float* B, float* C, size_t pitch, int width){

//compute indexes
int col = blockIdx.x * blockDim.x + threadIdx.x;


int rowWidthWithPad = pitch/sizeof(float);

if(col < width){
for (int row = 0; row < width; ++row) {
if(row < width)
C[row * rowWidthWithPad  + col] = A[row * rowWidthWithPad  + col] + B[row * rowWidthWithPad  + col];
}
}


}