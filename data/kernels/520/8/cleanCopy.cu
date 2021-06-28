#include "includes.h"
__global__ void cleanCopy(int *S, int *D){
D[threadIdx.x] = S[threadIdx.x];
}