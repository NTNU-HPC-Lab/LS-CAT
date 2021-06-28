#include "includes.h"
__global__ void vecAdd(unsigned int *A_d, unsigned int *B_d, unsigned int *C_d, int WORK_SIZE) {

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// **** Populate vecADD kernel function ****
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// Get our global thread ID
int id = blockIdx.x*blockDim.x+threadIdx.x;

// Make sure we do not go out of bounds
if (id < WORK_SIZE)
C_d[id] = A_d[id] + B_d[id];


}