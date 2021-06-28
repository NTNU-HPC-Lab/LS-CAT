#include "includes.h"
__global__ void matrixAdd_B_Kernel(float* A, float* B, float* C, size_t pitch, int width){

//compute indexes
int row = blockIdx.x * blockDim.x + threadIdx.x;


int rowWidthWithPad = pitch/sizeof(float);

if(row < width){
for (int col = 0; col < width; ++col) {
if(col < width)
C[row * rowWidthWithPad  + col] = A[row * rowWidthWithPad  + col] + B[row * rowWidthWithPad  + col];
}
}


}