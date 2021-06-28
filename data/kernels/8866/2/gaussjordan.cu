#include "includes.h"
__global__ void gaussjordan(double *A, double *I, int n, int i) {
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < n && y < n) {
if (x != i) {
I[x * n + y] -= I[i * n + y] * A[x * n + i];
if (y != i) {
A[x * n + y] -= A[i * n + y] * A[x * n + i];
}
}
}
}