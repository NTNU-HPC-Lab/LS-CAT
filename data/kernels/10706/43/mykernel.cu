#include "includes.h"
__global__ void mykernel(float *d1, float *d2, float *d3, float *d4, float *d5) {
if(threadIdx.x == 0) {
d1[0] = 123.0f;
d2[0] = 123.0f;
d3[0] = 123.0f;
d4[0] = 123.0f;
d5[0] = 123.0f;
}
}