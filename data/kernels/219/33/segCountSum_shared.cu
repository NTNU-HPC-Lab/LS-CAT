#include "includes.h"
__global__ void segCountSum_shared(int *counter, int *segcounter, const int countlength)
{
unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
extern __shared__ int s_counter[];

if (xIndex < countlength){
for (int jj=0; jj<countlength; jj++){
s_counter[xIndex] = s_counter[xIndex] + segcounter[xIndex + jj*countlength];
}
}
counter[xIndex] = s_counter[xIndex];
}