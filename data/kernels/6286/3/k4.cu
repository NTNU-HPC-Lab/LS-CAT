#include "includes.h"
__global__ void k4(int *Aux,int *S){
if(blockIdx.x==0) return;
int tid=blockIdx.x*B+threadIdx.x;
S[tid]+=Aux[blockIdx.x-1];
}