#include "includes.h"
__device__ int2 devInt2[10];  __global__ void regOperation() {
int2 f = devInt2[1];
devInt2[0] = f;
}
__global__ void regOperation2() {
int2 f = devInt2[1];
devInt2[0].x = f.x;
devInt2[0].y = f.y;
}