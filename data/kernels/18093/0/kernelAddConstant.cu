#include "includes.h"
#define CUDAMAXTHREADPERBLOCK 1024
#define CUDAMAXBLOCK 65536

using namespace std;

__global__ void kernelAddConstant(int *g_a, const int b)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
g_a[idx] += b;
}