#include "includes.h"
__global__ void externUndoAssignParallel(int* domain, int size, int value){

if(threadIdx.x + blockIdx.x * blockDim.x < size &&
threadIdx.x + blockIdx.x * blockDim.x != value)
++domain[threadIdx.x + blockIdx.x * blockDim.x];

}