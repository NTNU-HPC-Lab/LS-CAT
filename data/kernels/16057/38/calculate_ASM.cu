#include "includes.h"
__global__ void calculate_ASM(float *norm,float *ASM,float *mulMatrix,int Max){
//printf("%d\n",max);
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;

// int Index = iy * N + ix;

for (int k = 0; k < Max; k++) {
// Accumulate results for a single element
// c[row * N + col] += a[row * N + k] * b[k * N + col];
// printf("C[%d] = a[%d] * b[%d]\n",row * N + col,row * N + k, k * N + col);
atomicAdd(&mulMatrix[row * Max + col],norm[row * Max + k] * norm[k * Max + col]);
}
int Index = blockIdx.x * blockDim.x + threadIdx.x;

atomicAdd(&ASM[0],mulMatrix[Index]);

if (Index == 0){

printf("ASM %f\n",ASM[0]);
}
}