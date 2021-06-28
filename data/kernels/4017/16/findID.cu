#include "includes.h"
__global__ void findID(double *a, int n){

// First we need to find our global threadID
int tPosX = blockIdx.x * blockDim.x + threadIdx.x;
// Make sure we are not out of range
if (tPosX < n){
a[tPosX] = tPosX;
}
}