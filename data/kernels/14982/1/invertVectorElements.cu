#include "includes.h"
extern "C"
__global__ void invertVectorElements(float* vector, int n)
{
int i = threadIdx.x;
if (i < n)
{
vector[i] = 1.0f / vector[i];
}
}