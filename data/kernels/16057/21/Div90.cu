#include "includes.h"
__global__ void Div90(int *matrix , int *newMatrix,int nx,int ny,int Max){
int ix = blockIdx.x * blockDim.x + threadIdx.x;
int iy = blockIdx.y * blockDim.y + threadIdx.y;

int Index = iy * nx + ix;
int posisi = 0;

for(int i = 0 ; i < nx - 1 ; ++i){
if(Index >= i * nx && Index < ((i + 1) * nx) - 1){
if(Index == 0 || Index % 2 == 0){
posisi = matrix[Index + nx] * nx + matrix[Index];
atomicAdd(&newMatrix[posisi],1);

posisi = matrix[Index + (nx + 1)] * nx + matrix[Index + 1];
atomicAdd(&newMatrix[posisi],1);
printf("Index : %d %d dan %d %d\n",Index + nx , Index, Index + (nx + 1),Index + 1);
}
}
}
}