#include "includes.h"
__global__ void add2(int a, int b, int *sum)
{
*sum = *sum + a + b;
}