#include "includes.h"
__global__ void addSingleThread(int n, float *x, float *y)
{
for (int i = 0; i < n; i++)
y[i] = x[i] + y[i];
}