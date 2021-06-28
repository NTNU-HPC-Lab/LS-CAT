#include "includes.h"
__global__ void Div180(int *matrix , int *newMatrix,int nx,int ny,int Max){
int ix = blockIdx.x * blockDim.x + threadIdx.x;
int iy = blockIdx.y * blockDim.y + threadIdx.y;

int Index = iy * nx + ix;
int posisi = 0;

for(int i = 0 ; i < nx ; i += 2){
if(Index >= i * nx && Index < ((i + 1) * nx) - 1){

posisi = matrix[Index + 1] * nx + matrix[Index];
atomicAdd(&newMatrix[posisi],1);

posisi = matrix[Index + (nx + 1)] * nx + matrix[Index + nx];
atomicAdd(&newMatrix[posisi],1);
}
}
}