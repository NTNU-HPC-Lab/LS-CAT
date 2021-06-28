#include "includes.h"
__global__ void div(float *a, float *b, float *c)
{
*c = *a / *b;
}