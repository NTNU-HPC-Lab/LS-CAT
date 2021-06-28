#include "includes.h"
__global__ void k5(int *Aux,int *S){
if(threadIdx.x==0) return;
S[(threadIdx.x+1)*B-1]=Aux[threadIdx.x];
}