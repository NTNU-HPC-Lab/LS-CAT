#include "includes.h"
__global__ void test_launch_cuda_native(float * scalar, float * vector, int sxy, int sx , int sy , int sz , int stride)
{
int id[3];

id[0] = threadIdx.x + blockIdx.x * blockDim.x;
id[1] = threadIdx.y + blockIdx.y * blockDim.y;
id[2] = threadIdx.z + blockIdx.z * blockDim.z;

if (id[0] >= sx) {return;}
if (id[1] >= sy) {return;}
if (id[2] >= sz) {return;}

float s = scalar[id[2]*sxy+id[1]*sx+id[0]];

float v[3];

v[0] = vector[id[2]*sxy+id[1]*sx+id[0] + 0*stride];
v[1] = vector[id[2]*sxy+id[1]*sx+id[0] + 1*stride];
v[2] = vector[id[2]*sxy+id[1]*sx+id[0] + 2*stride];

printf("Grid point from CUDA %d %d %d     scalar: %f  vector: %f %f %f \n",id[0],id[1],id[2],s,v[0],v[1],v[2]);
}