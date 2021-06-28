#include "includes.h"
__device__ int f () { return 21; }
__global__ void sample()
{
int a = blockIdx.x;
int b = blockIdx.y;
int c = threadIdx.x;
double x = 1;

double result = pow(0.0,x)+a+b*x+c*pow(x,2.0);

if(result == 10)
printf("a=%d, b=%d, c=%d\n", a,b,c);
}