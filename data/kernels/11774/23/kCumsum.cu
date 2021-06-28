#include "includes.h"
__global__ void kCumsum(float *mat, float *target, float *temp, unsigned int height) {
// extern  __shared__  float temp[];// allocated on invocation
const int thid = threadIdx.x;
if (2*thid < height) {
const int super_offset = blockIdx.x * height;
target += super_offset;
mat += super_offset;
temp += super_offset;
int offset = 1;
//float s = 0.0;
temp[2*thid]   = mat[2*thid]; // load input into shared memory
temp[2*thid+1] = mat[2*thid+1];
for (int d = height>>1; d > 0; d >>= 1) {// build sum in place up the tree
__syncthreads();
if (thid < d) {
int ai = offset*(2*thid+1)-1;
int bi = offset*(2*thid+2)-1;
temp[bi] += temp[ai];
} else if (thid == d && thid % 2 == 1) {
//int bi = offset*(2*thid+2)-1;
//temp[bi] += temp[ai];

}

offset *= 2;
}
if (thid == 0) { temp[height - 1] = 0; } // clear the last element
for (int d = 1; d < height; d *= 2)  { // traverse down tree & build scan
offset >>= 1;
__syncthreads();
if (thid < d) {
int ai = offset*(2*thid+1)-1;
int bi = offset*(2*thid+2)-1;
float t   = temp[ai];
temp[ai]  = temp[bi];
temp[bi] += t;
}
}
__syncthreads();
// write results to device memory
//  if (thid == -1) {
//    target[0]   = temp[1];
//    target[height-1] = s;
//  } else {
target[2*thid]   = temp[2*thid];
target[2*thid+1] = temp[2*thid+1];
//  }
}

}