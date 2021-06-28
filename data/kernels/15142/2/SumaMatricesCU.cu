#include "includes.h"
__global__ void SumaMatricesCU(int* A,int* B,int* C,int width){
int col=blockIdx.x*blockDim.x + threadIdx.x;//columnas
int row=blockIdx.y*blockDim.y + threadIdx.y;//filas

if((row<width)&&(col<width)){
C[row*width+col] = A[row*width+col]+B[row*width+col];
}
}