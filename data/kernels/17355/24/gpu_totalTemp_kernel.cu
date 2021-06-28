#include "includes.h"

__global__ void gpu_totalTemp_kernel ( int N, double * partialT, double * totalT)
{
extern __shared__ double T_cache[];
int tid = threadIdx.x;

T_cache[tid] = partialT[tid];

__syncthreads();

int nTotalThreads = blockDim.x;               /// Total number of active threads

/** Algoritme per calcular la reduccio
*  dels valors actuals a la cache del block */
while(nTotalThreads > 1)
{
int halfPoint = (nTotalThreads >> 1);       /// divide by two, only the first half of the threads will be active.

if (threadIdx.x < halfPoint)
T_cache[threadIdx.x] += T_cache[threadIdx.x + halfPoint];

__syncthreads();                /// imprescindible

nTotalThreads = halfPoint;      /// Reducing the binary tree size by two:
}


/// El primer thread de cada block es el k s'encarrega de fer els calculs finals
if(threadIdx.x == 0) {

double T = T_cache[0];

T /= (kb * dim * N);  /// Instantaneous temperature using the Equipartition Theorem. The kinetic energy is just K = 3N/2 kT

(*totalT) = T;
}
}