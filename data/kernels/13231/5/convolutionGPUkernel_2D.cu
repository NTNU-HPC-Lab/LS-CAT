#include "includes.h"
__global__ void  convolutionGPUkernel_2D(int *M, int *mascara,int *resultado,int m, int n, int widthM){
int col = blockIdx.x*blockDim.x + threadIdx.x;
int row = blockIdx.y*blockDim.y + threadIdx.y;

if(col < n && row < m){
int p = 0;
int start_col = col - (widthM/2);
int start_row = row - (widthM/2);
for (int i = 0; i < widthM ; i++) {
for (int j = 0; j < widthM; j++) {
int curRow = start_row + i;
int curCol = start_col + j;
if(curRow > -1 && curRow < m && curCol > -1 && curCol < n){
p += M[curRow*m + curCol]*mascara[i*widthM + j];
}
}
}
resultado[row*n + col] = p;
}
}