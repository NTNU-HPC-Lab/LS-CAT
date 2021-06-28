#include "includes.h"
__global__ void grayscaleKernel(int *ms, int *aux, int n){
int i = threadIdx.x+blockDim.x*blockIdx.x;
int k=0;

int grayscale=0;
if(i<n){
for(k=0; k<n-3; k+=3){
grayscale = 0.299*ms[i*n+k] + 0.5876*ms[i*n+k+1] + 0.114*ms[i*n+k+2];
aux[i*n+k] = aux[i*n+k+1] = aux[i*n+k+2] = grayscale;
}
}
}