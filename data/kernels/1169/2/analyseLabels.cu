#include "includes.h"
/*
* CCL3D.cu
*/


#define CCL_BLOCK_SIZE_X 8
#define CCL_BLOCK_SIZE_Y 8
#define CCL_BLOCK_SIZE_Z 8

__device__ int d_isNotDone;




__global__ void analyseLabels(int* labels, int w, int h, int d) {
const int x = blockIdx.x * CCL_BLOCK_SIZE_X + threadIdx.x;
const int y = blockIdx.y * CCL_BLOCK_SIZE_Y + threadIdx.y;
const int z = blockIdx.z * CCL_BLOCK_SIZE_Z + threadIdx.z;
const int index = (z*h + y)*w + x;

if (x >= w || y >= h || z >= d) return;

int lcur = labels[index];
if (lcur) {
int r = labels[lcur];
while(r != lcur) {
lcur = labels[r];
r = labels[lcur];
}
labels[index] = lcur;
}
}