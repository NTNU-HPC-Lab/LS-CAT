#include "includes.h"
__device__ __forceinline__ size_t gpu_fieldn_index(unsigned int x, unsigned int y, unsigned int z, unsigned int d)
{
return (NX*(NY*(NZ*(d-1)+z)+y)+x);
}
__global__ void gpu_stream(double *f0, double *f1, double *f2, double *h0, double *h1, double *h2, double *temp0, double *temp1, double *temp2)
{
unsigned int y = blockIdx.y;
unsigned int z = blockIdx.z;
unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

// streaming step
unsigned int xp1 = (x + 1) % NX;
unsigned int yp1 = (y + 1) % NY;
unsigned int zp1 = (z + 1) % NZ;
unsigned int xm1 = (NX + x - 1) % NX;
unsigned int ym1 = (NY + y - 1) % NY;
unsigned int zm1 = (NZ + z - 1) % NZ;
// direction numbering scheme
// 6 2 5
// 3 0 1
// 7 4 8

// load populations from adjacent nodes (ft is post-streaming population of f1)
// flows
f1[gpu_fieldn_index(x, y, z, 1)] = f2[gpu_fieldn_index(xm1, y, z, 1)];
f1[gpu_fieldn_index(x, y, z, 2)] = f2[gpu_fieldn_index(xp1, y, z, 2)];
f1[gpu_fieldn_index(x, y, z, 3)] = f2[gpu_fieldn_index(x, ym1, z, 3)];
f1[gpu_fieldn_index(x, y, z, 4)] = f2[gpu_fieldn_index(x, yp1, z, 4)];
f1[gpu_fieldn_index(x, y, z, 5)] = f2[gpu_fieldn_index(x, y, zm1, 5)];
f1[gpu_fieldn_index(x, y, z, 6)] = f2[gpu_fieldn_index(x, y, zp1, 6)];
f1[gpu_fieldn_index(x, y, z, 7)] = f2[gpu_fieldn_index(xm1, ym1, z, 7)];
f1[gpu_fieldn_index(x, y, z, 8)] = f2[gpu_fieldn_index(xp1, yp1, z, 8)];
f1[gpu_fieldn_index(x, y, z, 9)] = f2[gpu_fieldn_index(xm1, y, zm1, 9)];
f1[gpu_fieldn_index(x, y, z, 10)] = f2[gpu_fieldn_index(xp1, y, zp1, 10)];
f1[gpu_fieldn_index(x, y, z, 11)] = f2[gpu_fieldn_index(x, ym1, zm1, 11)];
f1[gpu_fieldn_index(x, y, z, 12)] = f2[gpu_fieldn_index(x, yp1, zp1, 12)];
f1[gpu_fieldn_index(x, y, z, 13)] = f2[gpu_fieldn_index(xm1, yp1, z, 13)];
f1[gpu_fieldn_index(x, y, z, 14)] = f2[gpu_fieldn_index(xp1, ym1, z, 14)];
f1[gpu_fieldn_index(x, y, z, 15)] = f2[gpu_fieldn_index(xm1, y, zp1, 15)];
f1[gpu_fieldn_index(x, y, z, 16)] = f2[gpu_fieldn_index(xp1, y, zm1, 16)];
f1[gpu_fieldn_index(x, y, z, 17)] = f2[gpu_fieldn_index(x, ym1, zp1, 17)];
f1[gpu_fieldn_index(x, y, z, 18)] = f2[gpu_fieldn_index(x, yp1, zm1, 18)];
f1[gpu_fieldn_index(x, y, z, 19)] = f2[gpu_fieldn_index(xm1, ym1, zm1, 19)];
f1[gpu_fieldn_index(x, y, z, 20)] = f2[gpu_fieldn_index(xp1, yp1, zp1, 20)];
f1[gpu_fieldn_index(x, y, z, 21)] = f2[gpu_fieldn_index(xm1, ym1, zp1, 21)];
f1[gpu_fieldn_index(x, y, z, 22)] = f2[gpu_fieldn_index(xp1, yp1, zm1, 22)];
f1[gpu_fieldn_index(x, y, z, 23)] = f2[gpu_fieldn_index(xm1, yp1, zm1, 23)];
f1[gpu_fieldn_index(x, y, z, 24)] = f2[gpu_fieldn_index(xp1, ym1, zp1, 24)];
f1[gpu_fieldn_index(x, y, z, 25)] = f2[gpu_fieldn_index(xp1, ym1, zm1, 25)];
f1[gpu_fieldn_index(x, y, z, 26)] = f2[gpu_fieldn_index(xm1, yp1, zp1, 26)];

// charges
h1[gpu_fieldn_index(x, y, z, 1)] = h2[gpu_fieldn_index(xm1, y, z, 1)];
h1[gpu_fieldn_index(x, y, z, 2)] = h2[gpu_fieldn_index(xp1, y, z, 2)];
h1[gpu_fieldn_index(x, y, z, 3)] = h2[gpu_fieldn_index(x, ym1, z, 3)];
h1[gpu_fieldn_index(x, y, z, 4)] = h2[gpu_fieldn_index(x, yp1, z, 4)];
h1[gpu_fieldn_index(x, y, z, 5)] = h2[gpu_fieldn_index(x, y, zm1, 5)];
h1[gpu_fieldn_index(x, y, z, 6)] = h2[gpu_fieldn_index(x, y, zp1, 6)];
h1[gpu_fieldn_index(x, y, z, 7)] = h2[gpu_fieldn_index(xm1, ym1, z, 7)];
h1[gpu_fieldn_index(x, y, z, 8)] = h2[gpu_fieldn_index(xp1, yp1, z, 8)];
h1[gpu_fieldn_index(x, y, z, 9)] = h2[gpu_fieldn_index(xm1, y, zm1, 9)];
h1[gpu_fieldn_index(x, y, z, 10)] = h2[gpu_fieldn_index(xp1, y, zp1, 10)];
h1[gpu_fieldn_index(x, y, z, 11)] = h2[gpu_fieldn_index(x, ym1, zm1, 11)];
h1[gpu_fieldn_index(x, y, z, 12)] = h2[gpu_fieldn_index(x, yp1, zp1, 12)];
h1[gpu_fieldn_index(x, y, z, 13)] = h2[gpu_fieldn_index(xm1, yp1, z, 13)];
h1[gpu_fieldn_index(x, y, z, 14)] = h2[gpu_fieldn_index(xp1, ym1, z, 14)];
h1[gpu_fieldn_index(x, y, z, 15)] = h2[gpu_fieldn_index(xm1, y, zp1, 15)];
h1[gpu_fieldn_index(x, y, z, 16)] = h2[gpu_fieldn_index(xp1, y, zm1, 16)];
h1[gpu_fieldn_index(x, y, z, 17)] = h2[gpu_fieldn_index(x, ym1, zp1, 17)];
h1[gpu_fieldn_index(x, y, z, 18)] = h2[gpu_fieldn_index(x, yp1, zm1, 18)];
h1[gpu_fieldn_index(x, y, z, 19)] = h2[gpu_fieldn_index(xm1, ym1, zm1, 19)];
h1[gpu_fieldn_index(x, y, z, 20)] = h2[gpu_fieldn_index(xp1, yp1, zp1, 20)];
h1[gpu_fieldn_index(x, y, z, 21)] = h2[gpu_fieldn_index(xm1, ym1, zp1, 21)];
h1[gpu_fieldn_index(x, y, z, 22)] = h2[gpu_fieldn_index(xp1, yp1, zm1, 22)];
h1[gpu_fieldn_index(x, y, z, 23)] = h2[gpu_fieldn_index(xm1, yp1, zm1, 23)];
h1[gpu_fieldn_index(x, y, z, 24)] = h2[gpu_fieldn_index(xp1, ym1, zp1, 24)];
h1[gpu_fieldn_index(x, y, z, 25)] = h2[gpu_fieldn_index(xp1, ym1, zm1, 25)];
h1[gpu_fieldn_index(x, y, z, 26)] = h2[gpu_fieldn_index(xm1, yp1, zp1, 26)];

// temperature
temp1[gpu_fieldn_index(x, y, z, 1)] = temp2[gpu_fieldn_index(xm1, y, z, 1)];
temp1[gpu_fieldn_index(x, y, z, 2)] = temp2[gpu_fieldn_index(xp1, y, z, 2)];
temp1[gpu_fieldn_index(x, y, z, 3)] = temp2[gpu_fieldn_index(x, ym1, z, 3)];
temp1[gpu_fieldn_index(x, y, z, 4)] = temp2[gpu_fieldn_index(x, yp1, z, 4)];
temp1[gpu_fieldn_index(x, y, z, 5)] = temp2[gpu_fieldn_index(x, y, zm1, 5)];
temp1[gpu_fieldn_index(x, y, z, 6)] = temp2[gpu_fieldn_index(x, y, zp1, 6)];
temp1[gpu_fieldn_index(x, y, z, 7)] = temp2[gpu_fieldn_index(xm1, ym1, z, 7)];
temp1[gpu_fieldn_index(x, y, z, 8)] = temp2[gpu_fieldn_index(xp1, yp1, z, 8)];
temp1[gpu_fieldn_index(x, y, z, 9)] = temp2[gpu_fieldn_index(xm1, y, zm1, 9)];
temp1[gpu_fieldn_index(x, y, z, 10)] = temp2[gpu_fieldn_index(xp1, y, zp1, 10)];
temp1[gpu_fieldn_index(x, y, z, 11)] = temp2[gpu_fieldn_index(x, ym1, zm1, 11)];
temp1[gpu_fieldn_index(x, y, z, 12)] = temp2[gpu_fieldn_index(x, yp1, zp1, 12)];
temp1[gpu_fieldn_index(x, y, z, 13)] = temp2[gpu_fieldn_index(xm1, yp1, z, 13)];
temp1[gpu_fieldn_index(x, y, z, 14)] = temp2[gpu_fieldn_index(xp1, ym1, z, 14)];
temp1[gpu_fieldn_index(x, y, z, 15)] = temp2[gpu_fieldn_index(xm1, y, zp1, 15)];
temp1[gpu_fieldn_index(x, y, z, 16)] = temp2[gpu_fieldn_index(xp1, y, zm1, 16)];
temp1[gpu_fieldn_index(x, y, z, 17)] = temp2[gpu_fieldn_index(x, ym1, zp1, 17)];
temp1[gpu_fieldn_index(x, y, z, 18)] = temp2[gpu_fieldn_index(x, yp1, zm1, 18)];
temp1[gpu_fieldn_index(x, y, z, 19)] = temp2[gpu_fieldn_index(xm1, ym1, zm1, 19)];
temp1[gpu_fieldn_index(x, y, z, 20)] = temp2[gpu_fieldn_index(xp1, yp1, zp1, 20)];
temp1[gpu_fieldn_index(x, y, z, 21)] = temp2[gpu_fieldn_index(xm1, ym1, zp1, 21)];
temp1[gpu_fieldn_index(x, y, z, 22)] = temp2[gpu_fieldn_index(xp1, yp1, zm1, 22)];
temp1[gpu_fieldn_index(x, y, z, 23)] = temp2[gpu_fieldn_index(xm1, yp1, zm1, 23)];
temp1[gpu_fieldn_index(x, y, z, 24)] = temp2[gpu_fieldn_index(xp1, ym1, zp1, 24)];
temp1[gpu_fieldn_index(x, y, z, 25)] = temp2[gpu_fieldn_index(xp1, ym1, zm1, 25)];
temp1[gpu_fieldn_index(x, y, z, 26)] = temp2[gpu_fieldn_index(xm1, yp1, zp1, 26)];
}