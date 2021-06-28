#include "includes.h"

# define M 10000
# define N 10000


__global__ void add( int * a, int * b, int * c)
{
unsigned int i= blockDim.x *blockIdx.x + threadIdx.x;
unsigned int j= blockDim.y *blockIdx.y + threadIdx.y;
if(i<M && j<N)
c[i*M+j]=a[i*M+j]+b[i*M+j];
}