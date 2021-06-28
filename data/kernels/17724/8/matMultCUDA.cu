#include "includes.h"
__global__ void matMultCUDA(const float* a, int lda, const float* b, int ldb, float* c, int ldc, int n)
{
__shared__ float matA[BLOCK_SIZE][BLOCK_SIZE];
__shared__ float matB[BLOCK_SIZE][BLOCK_SIZE];
const int tidc = threadIdx.x;
const int tidr = threadIdx.y;
const int bidc = blockIdx.x*BLOCK_SIZE;
const int bidr = blockIdx.y*BLOCK_SIZE;
int i,j;
float results = 0;
float comp = 0;
for (j=0; j<n; j+=BLOCK_SIZE) {
matA[tidr][tidc] = a[(tidr+bidr)*lda+tidc+j];
matB[tidr][tidc] = b[(tidr+j)*ldb+tidc+bidc];

__syncthreads();

for (i=0;i<BLOCK_SIZE;i++){
float t;
comp -= matA[tidr][i]*matB[i][tidc];
t=results-comp;
comp=(t-results)+comp;
results=t;
}

__syncthreads();
}
c[(tidr+bidr)*ldc+tidc+bidc]=results;
}