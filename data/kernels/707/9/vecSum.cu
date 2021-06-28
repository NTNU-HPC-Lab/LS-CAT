#include "includes.h"
__global__ void vecSum(double* devIn, int pow_step, int n)
{
//The thread ID (including its block ID)
int i = blockIdx.x * blockDim.x + threadIdx.x;

//Safety check to prevent unwanted threads.
if(pow_step*i < n)
//The two 'adjacent' elements of the array (or
//the two children in the segment tree) are added and
//the result is stored in the first element.
devIn[pow_step*i] = devIn[pow_step*i+(pow_step/2)] + devIn[pow_step*i];
}