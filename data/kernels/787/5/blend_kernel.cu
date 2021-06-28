#include "includes.h"
extern "C" {
}



#define TB 256
#define EPS 0.1

#undef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#undef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))


__global__ void blend_kernel( float *A, float *BP, float *M, float *AP, float alpha, int c, int h, int w )
{
int _id = blockIdx.x * blockDim.x + threadIdx.x;
int size = h * w;
if (_id < c * size) {
// _id = dc * size + id
int id = _id % size, dc = _id / size;
// int x = id % w, y = id / w;
float weight = M[id] < 0.05f ? 0.f : alpha;
AP[dc * size + id] =
A[dc * size + id] * weight +
BP[dc * size + id] * (1.f - weight);
}
return ;
}