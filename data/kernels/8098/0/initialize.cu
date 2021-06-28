#include "includes.h"


#define THREADS_PER_BLOCK 1024
#define TIME 3600000








__global__ void initialize(float *a_d, float *b_d, float *c_d, int arraySize)
{
int ix = blockIdx.x * blockDim.x + threadIdx.x;
if(ix==0)
{
a_d[ix]=200.0;
b_d[ix]=200.0;

}

else if (ix<arraySize)
{
a_d[ix]=0.0;
b_d[ix]=0.0;
}

}