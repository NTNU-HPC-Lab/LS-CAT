#include "includes.h"
__global__ void mul(int *a, int *b, int *c)
{
*c = *a * *b;
}