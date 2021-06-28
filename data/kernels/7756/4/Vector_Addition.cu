#include "includes.h"
__global__ void Vector_Addition ( const int *dev_a , const int *dev_b , int *dev_c)
{
//Get the id of thread within a block
unsigned short tid = blockDim.x*blockIdx.x+threadIdx.x;

if ( tid < N ) // check the boundry condition for the threads
dev_c [tid] = dev_a[tid] + dev_b[tid] ;

}