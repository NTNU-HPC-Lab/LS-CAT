#include "includes.h"




using namespace std;

__global__ void matrixEuclideanDistanceKernelFastPartialOut(float* in_X, float* in_Y, float* out, int_least64_t n, int_least64_t m, int_least64_t start_out, int_least64_t end_out){
__shared__ float Ys[16][16];
__shared__ float Xs[16][16];

int_least64_t bx = blockIdx.x, by = blockIdx.y;
int_least64_t tx = threadIdx.x, ty = threadIdx.y;

int_least64_t yBegin = by * 16 * m;
int_least64_t xBegin = bx * 16 * m;

int_least64_t yEnd = yBegin + m - 1, y, x, k;
int_least64_t o;

float tmp, s = 0;

for (y = yBegin, x = xBegin;
y <= yEnd;
y += 16, x += 16){
Ys[ty][tx] = in_Y[y + ty * m + tx];
Xs[tx][ty] = in_X[x + ty * m + tx];
__syncthreads();

for (k = 0; k<16; k++){
tmp = Ys[ty][k] - Xs[k][tx];
s += tmp * tmp;
}
__syncthreads();
}

o = by * 16 * n + ty * n + bx * 16 + tx;
if (o >= start_out && o < end_out){
out[o - start_out] = s;
}
}