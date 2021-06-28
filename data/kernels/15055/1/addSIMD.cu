#include "includes.h"



#define PARTSIZE 4



__global__ void addSIMD(unsigned int *data1, unsigned int *data2)
{
int thread = (gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x + threadIdx.x;
unsigned int *ptr1 = data1 + thread;
unsigned int *ptr2 = data2 + thread;

*ptr1 = __vaddus4(*ptr1, *ptr2);
}