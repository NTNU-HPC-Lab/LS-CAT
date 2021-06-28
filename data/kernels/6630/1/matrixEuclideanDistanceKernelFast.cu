#include "includes.h"




using namespace std;

__global__ void matrixEuclideanDistanceKernelFast(float* in, float* out, int n, int m){
__shared__ float Ys[16][16];
__shared__ float Xs[16][16];

int bx = blockIdx.x, by = blockIdx.y;
int tx = threadIdx.x, ty = threadIdx.y;

int yBegin = by * 16 * m;
int xBegin = bx * 16 * m;

int yEnd = yBegin + m - 1, y, x, k, o;

float tmp, s = 0;

for (y = yBegin, x = xBegin;
y <= yEnd;
y += 16, x += 16){
Ys[ty][tx] = in[y + ty * m + tx];
Xs[tx][ty] = in[x + ty * m + tx];
__syncthreads();

for (k = 0; k<16; k++){
tmp = Ys[ty][k] - Xs[k][tx];
s += tmp * tmp;
}
__syncthreads();
}
o = by * 16 * n + ty * n + bx * 16 + tx;
out[o] = s;
}