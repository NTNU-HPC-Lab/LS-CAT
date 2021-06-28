#include "includes.h"
__global__ void smooth(float * v_new, const float * v) {
int myIdx = threadIdx.x * gridDim.x + blockIdx.x;
int numThreads = blockDim.x * gridDim.x;
int myLeftIdx = (myIdx == 0) ? 0 : myIdx - 1;
int myRightIdx = (myIdx == (numThreads - 1)) ? numThreads - 1 : myIdx + 1;
float myElt = v[myIdx];
float myLeftElt = v[myLeftIdx];
float myRightElt = v[myRightIdx];
v_new[myIdx] = 0.25f * myLeftElt + 0.5f * myElt + 0.25f * myRightElt;
}