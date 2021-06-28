#include "includes.h"
__global__ void VectorAdd(int *a, int *r, int n, double gamma)
{
int i=threadIdx.x;

if(i<n)
r[i] = (int)(255.0*pow((double)a[i]/255.0,1.0/gamma));
}