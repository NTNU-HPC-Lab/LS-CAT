#include "includes.h"
__device__ __forceinline__ size_t gpu_fieldn_index(unsigned int x, unsigned int y, unsigned int d)
{
return (NX*(NY*(d-1)+y)+x);
}
__global__ void gpu_stream(double *f0, double *f1, double *f2, double *h0, double *h1, double *h2)
{
unsigned int y = blockIdx.y;
unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

// streaming step

unsigned int xp1 = (x + 1) % NX;
unsigned int yp1 = (y + 1) % NY;
unsigned int xm1 = (NX + x - 1) % NX;
unsigned int ym1 = (NY + y - 1) % NY;

// direction numbering scheme
// 6 2 5
// 3 0 1
// 7 4 8

// load populations from adjacent nodes (ft is post-streaming population of f1)
f1[gpu_fieldn_index(x, y, 1)] = f2[gpu_fieldn_index(xm1, y, 1)];
f1[gpu_fieldn_index(x, y, 2)] = f2[gpu_fieldn_index(x, ym1, 2)];
f1[gpu_fieldn_index(x, y, 3)] = f2[gpu_fieldn_index(xp1, y, 3)];
f1[gpu_fieldn_index(x, y, 4)] = f2[gpu_fieldn_index(x, yp1, 4)];
f1[gpu_fieldn_index(x, y, 5)] = f2[gpu_fieldn_index(xm1, ym1, 5)];
f1[gpu_fieldn_index(x, y, 6)] = f2[gpu_fieldn_index(xp1, ym1, 6)];
f1[gpu_fieldn_index(x, y, 7)] = f2[gpu_fieldn_index(xp1, yp1, 7)];
f1[gpu_fieldn_index(x, y, 8)] = f2[gpu_fieldn_index(xm1, yp1, 8)];

h1[gpu_fieldn_index(x, y, 1)] = h2[gpu_fieldn_index(xm1, y, 1)];
h1[gpu_fieldn_index(x, y, 2)] = h2[gpu_fieldn_index(x, ym1, 2)];
h1[gpu_fieldn_index(x, y, 3)] = h2[gpu_fieldn_index(xp1, y, 3)];
h1[gpu_fieldn_index(x, y, 4)] = h2[gpu_fieldn_index(x, yp1, 4)];
h1[gpu_fieldn_index(x, y, 5)] = h2[gpu_fieldn_index(xm1, ym1, 5)];
h1[gpu_fieldn_index(x, y, 6)] = h2[gpu_fieldn_index(xp1, ym1, 6)];
h1[gpu_fieldn_index(x, y, 7)] = h2[gpu_fieldn_index(xp1, yp1, 7)];
h1[gpu_fieldn_index(x, y, 8)] = h2[gpu_fieldn_index(xm1, yp1, 8)];
}