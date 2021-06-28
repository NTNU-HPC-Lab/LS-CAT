#include "includes.h"
__global__ void add(int *a, int *b,int * c)
{
int col=10;
int i= blockIdx.y*blockDim.y+threadIdx.y;
int j=blockIdx.x*blockDim.x+threadIdx.x;


*(c + i * col +j)= *(a + i * col + j) + *(b + i * col + j);

}