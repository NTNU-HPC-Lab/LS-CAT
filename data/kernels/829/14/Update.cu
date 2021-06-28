#include "includes.h"
__global__ void Update(float *WHAT , float *WITH , float AMOUNT) {
int idx = threadIdx.x + blockIdx.x * blockDim.x; // which voxel
WHAT[idx] +=AMOUNT*WITH[idx];
}