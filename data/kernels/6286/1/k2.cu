#include "includes.h"
__global__ void k2(int *Aux,int *S){

Aux[threadIdx.x]=S[(threadIdx.x+1)*B-1];
}