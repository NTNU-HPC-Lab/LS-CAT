#include "includes.h"
__global__ void Vector_Addition (  int *dev_a ,  int *dev_b , int *dev_c)
{
//Lay ra id cua thread trong 1 block.
int tid = blockIdx.x ; // blockDim.x*blockIdx.x+threadIdx.x

if ( tid < N )
*(dev_c+tid) = *(dev_a+tid) + *(dev_b+tid) ;

}