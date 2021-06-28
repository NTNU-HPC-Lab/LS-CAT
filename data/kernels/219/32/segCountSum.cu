#include "includes.h"
__global__ void segCountSum(int *counter, int *segcounter, const int countlength)
{
unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;

if (xIndex < countlength){
for (int jj=0; jj<countlength; jj++){
counter[xIndex] = counter[xIndex] + segcounter[xIndex + jj*countlength];
}
}
}