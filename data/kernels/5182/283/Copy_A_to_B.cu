#include "includes.h"
__global__  void Copy_A_to_B (float * A , float * B , int size){
int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
if (id<size)
B[id] = A[id];
}