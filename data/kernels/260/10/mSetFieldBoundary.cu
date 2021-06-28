#include "includes.h"
__device__ bool checkBoundary(int blockIdx, int blockDim, int threadIdx){
int x = threadIdx;
int y = blockIdx;
return (x == 0 || x == (blockDim-1) || y == 0 || y == 479);
}
__global__ void mSetFieldBoundary(float *field, float scalar) {
if(checkBoundary(blockIdx.x, blockDim.x, threadIdx.x)) {
int Idx = blockIdx.x * blockDim.x + threadIdx.x;
int x = threadIdx.x;
int y = blockIdx.x;

if(x == 0 && y == 0) {
field[Idx] = field[Idx+blockDim.x+1]*scalar;
} else if(x == 0 && y == blockDim.x-1) {
field[Idx] = field[Idx-blockDim.x+1]*scalar;
} else if (x == blockDim.x-1 && y == 0) {
field[Idx] = field[Idx+blockDim.x-1]*scalar;
} else if (x == blockDim.x-1 && y == blockDim.x-1) {
field[Idx] = field[Idx-blockDim.x-1]*scalar;
} else if (x == 0) {
field[Idx] = field[Idx+1]*scalar;
} else if(x == blockDim.x-1) {
field[Idx] = field[Idx-1]*scalar;
} else if(y == 0) {
field[Idx] = field[Idx+blockDim.x]*scalar;
} else field[Idx] = field[Idx-blockDim.x]*scalar;
} else return;
}