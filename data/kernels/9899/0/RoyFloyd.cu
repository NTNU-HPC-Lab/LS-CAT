#include "includes.h"

#define INF 1000000

using namespace std;


__global__ void RoyFloyd(int* matrix, int k, int N)
{
int i = blockDim.y * blockIdx.y + threadIdx.y;
int j = blockDim.x * blockIdx.x + threadIdx.x;

if (matrix[i*N + k] + matrix[k*N + j] < matrix[i*N + j])
matrix[i*N + j] = matrix[i*N + k] + matrix[k*N + j];
}