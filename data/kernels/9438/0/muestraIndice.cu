#include "includes.h"

#define N 24


__global__ void muestraIndice(float *a, float *b, float *c){

int global = blockIdx.x * blockDim.x + threadIdx.x;

if(global < N){
a[global] = threadIdx.x;
b[global] = blockIdx.x;
c[global] = global;
}

}