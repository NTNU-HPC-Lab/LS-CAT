#include "includes.h"
__global__ void maxi(int * a, int * b, int n) {

int block = 256 * blockIdx.x;
int max = 0;

for (int i = block; i < min(256 + block, n); i++) {
if (max < a[i]) {
max = a[i];
}
}
b[blockIdx.x] = max;
}