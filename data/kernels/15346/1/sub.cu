#include "includes.h"
__global__ void sub(int *a, int *b, int *c)
{
*c = *a - *b;
}