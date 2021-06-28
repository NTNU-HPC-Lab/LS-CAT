#include "includes.h"
__global__ void addTwoIntegers(int *a, int *b, int *c)
{
//	int i = threadIdx.x;
*c = *a + *b;
}