#include "includes.h"
__global__ void LinearPolynomForwardImpl( const float* probs, int batchSize, const float* values, int polynomCount, int outputDim, float* out) {

//out: batch_elem0 dim0, dim1, dimk batch_elem1 dim0 dim1 dimk
//so threads
int polynomId = blockIdx.x;
const int dimId = blockIdx.y;

int tid = threadIdx.x;
if (tid >= batchSize) {
return;
}

float sum = 0;
probs += threadIdx.x;
values += dimId;

while (polynomId < polynomCount) {
const float polynomProb = __ldg(probs + polynomId * batchSize); // includes x
const float v = __ldg(values + polynomId * outputDim);
sum += polynomProb * v;
polynomId += gridDim.x;
}

atomicAdd(out + dimId * batchSize + threadIdx.x, sum);
}