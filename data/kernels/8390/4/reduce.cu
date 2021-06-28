#include "includes.h"
__global__ void reduce(int * vector,int size,int pot){

int idx = threadIdx.x + blockIdx.x*blockDim.x;
int salto = pot/2;

while(salto){
if(idx<salto && idx+salto<size){
vector[idx]=vector[idx]+vector[idx+salto];
}
__syncthreads();
salto=salto/2;
}

return;
}