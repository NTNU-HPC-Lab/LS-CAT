#include "includes.h"
__global__ void vecAddKernel(float *a, float *b, float *c, int n)
{
//ID del thread
int id = blockIdx.x*blockDim.x+threadIdx.x;


//No salir del tamaño del vector
if (id < n)
c[id] = a[id] + b[id];
}