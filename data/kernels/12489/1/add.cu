#include "includes.h"
__global__ void add( double *a, double *b, double *c, int n )
{
int tid = threadIdx.x + blockIdx.x * blockDim.x;
// handle the data at this index
while (tid < n) {
c[tid] = a[tid] + b[tid];
tid += blockDim.x * gridDim.x;
}
//printf("Value of *ip variable: %f\n", a[tid] );

}