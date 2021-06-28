#include "includes.h"
using namespace std;

// GPU Code
// __global__ indicates that it is a GPU kernel, that can be called from the CPU

// CPU Code
__global__ void Add(float* d_a, float* d_b, float* d_c, int N)
{
int id = blockIdx.x * blockDim.x + threadIdx.x;
if(id < N)

d_c[id] = d_a[id] + d_b[id];

}