#include "includes.h"
#define N 128*128



__global__ void kernelMontecarlo(float *x, float *y,int *contador) {
//int i = threadIdx.x + blockIdx.x*blockDim.x;
//int j = threadIdx.y + blockIdx.y*blockDim.y;
int indice = threadIdx.x + blockIdx.x*blockDim.x;
//int indice=i;
//printf("Indice: %f\n",(x[indice]*x[indice] + y[indice]*y[indice]));
if((x[indice]*x[indice] + y[indice]*y[indice]) <=1.0) {
atomicAdd(contador,1);//contador++;
//printf("Contador: %d\n",*contador);
}
}