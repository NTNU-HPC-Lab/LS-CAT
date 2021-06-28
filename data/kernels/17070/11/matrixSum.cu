#include "includes.h"
__global__ void matrixSum(int* a,int* b, int* c, int size)
{
// int max = maxThreadsPerBlock;
// printf("ERROR en global\n");
int pos = threadIdx.x + blockIdx.x * blockDim.x;
// printf("Block: %d\n", blockIdx.x );
// printf("pos= %d\n",pos);
if(pos<size*size){
c[pos] = a[pos] + b[pos];
}
}