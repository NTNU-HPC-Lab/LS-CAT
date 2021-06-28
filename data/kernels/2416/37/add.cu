#include "includes.h"
__global__ void add(int *a, int *b, int *sum)
{
*sum = *a + *b;
}