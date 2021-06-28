#include "includes.h"

using namespace std;

/* Utility function, use to do error checking.

Use this function like this:

checkCudaCall(cudaMalloc((void **) &deviceRGB, imgS * sizeof(color_t)));

And to check the result of a kernel invocation:

checkCudaCall(cudaGetLastError());
*/
__global__ void vectorTransformKernel(float* A, float* B, float* Result) {
// insert operation here
int i = threadIdx.x + blockDim.x * blockIdx.x;
if(i < 1000000)
Result[i] = Result[i] + A[i] * B[i];
}