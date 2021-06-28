#include "includes.h"
// CUDA-C includes



extern "C" void runCudaPart();




// Main cuda function

__global__ void addAry( int * ary1, int * ary2 )
{
int indx = threadIdx.x;
ary1[ indx ] += ary2[ indx ];
}