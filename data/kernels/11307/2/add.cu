#include "includes.h"
__global__ void add(int a, int b, int *c)
{
//Add 2 numbers together and store in location pointed by *c
*c = a + b;
}