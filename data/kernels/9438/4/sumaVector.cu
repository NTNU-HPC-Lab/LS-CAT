#include "includes.h"
__global__ void sumaVector(float *v1, float *v2, float *res){

int index = blockIdx.x * blockDim.x + threadIdx.x;

if(index < TAM_V)
res[index] = v1[index] + v2[index];

}