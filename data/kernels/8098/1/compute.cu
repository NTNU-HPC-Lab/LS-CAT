#include "includes.h"


#define THREADS_PER_BLOCK 1024
#define TIME 3600000








__global__ void compute(float *a_d, float *b_d, float *c_d, int arraySize)
{
int ix = blockIdx.x * blockDim.x + threadIdx.x;
float temp;
if( ix > 0 && ix < arraySize-1){
temp = (a_d[ix+1]+a_d[ix-1])/2.0;
__syncthreads();
b_d[ix]=temp;
__syncthreads();
}



}