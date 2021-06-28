#include "includes.h"
using namespace std;

#define GRID_SIZE 32
#define SHARED_MEM 16384


__global__ void findY(float *x, float *y, int n, float h, float z, int zLoc, float *returnVal) {
// int col = blockIdx.x * blockDim.x + threadIdx.x;
// int row = blockIdx.y * blockDim.y + threadIdx.y;

__shared__ float sum;
sum = 0;
// float absVal = 0;
int count = 0;
for(int i = 0; i < n; i++) {
// absVal = abs(x[i] - z);
if(abs(x[i] - z) < h) {
//sum = atomicAdd(&sum, y[zLoc]);
sum += y[i];
// cuPrintf("sum = %d\n", sum);
count++;
}
}
*returnVal = sum / count;
// sum = 0;
// count = 0;
}