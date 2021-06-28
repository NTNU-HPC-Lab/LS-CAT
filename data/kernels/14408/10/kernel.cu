#include "includes.h"
__global__ void kernel(int *A, int *B, int *counter, int n) {
int tid = threadIdx.x;

if (tid < n) {
for (int j=0; j<n; j++) {
counter[tid*n+j]++;
A[tid*n+j] = B[tid*n+j];
}
}
}