#include "includes.h"
__global__ void suma_GPU(int a, int b, int *c)
{
*c = a + b;
}