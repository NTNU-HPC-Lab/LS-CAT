#include "includes.h"
__global__ void cuda_mat_multiply(const double* A, const double* B, double * C, int rowsa, int colsa, int rowsb, int colsb, int rowsc, int colsc){
__shared__ double sA[32][32];   // Tile size of 32x32
__shared__ double sB[32][32];
int Row = blockDim.y*blockIdx.y + threadIdx.y;
int Col = blockDim.x*blockIdx.x + threadIdx.x;
double Cvalue = 0.0;
sA[threadIdx.y][threadIdx.x] = 0.0;
sB[threadIdx.y][threadIdx.x] = 0.0;
for (int k = 0; k < (((colsa - 1)/ 32) + 1); k++){
if ( (Row < rowsa) && (threadIdx.x + (k*32)) < colsa){
sA[threadIdx.y][threadIdx.x] = A[(Row*colsa) + threadIdx.x + (k*32)];
}
else{
sA[threadIdx.y][threadIdx.x] = 0.0;
}
__syncthreads();
if ( Col < colsb && (threadIdx.y + k*32) < rowsb){
sB[threadIdx.y][threadIdx.x] = B[(threadIdx.y + k*32)*colsb + Col];
}
else{
sB[threadIdx.y][threadIdx.x] = 0.0;
}
__syncthreads();

for (int j = 0; j < 32; ++j){
Cvalue += sA[threadIdx.y][j] * sB[j][threadIdx.x];
}
__syncthreads();
}
if (Row < rowsc && Col < colsc){
C[Row*colsc + Col] = Cvalue;
}
}