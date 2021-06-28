#include "includes.h"
__global__ void calcPrimes(int *d_IL, int *d_PL, int numOfPrimes, int lenInputList) {
int index = threadIdx.x + blockIdx.x * blockDim.x;
if(index < numOfPrimes) {
for(int i = d_PL[numOfPrimes-1]+1; i < lenInputList; i++) {
if(i % d_PL[index] == 0) {
d_IL[i] = 0;
}
}
}
}