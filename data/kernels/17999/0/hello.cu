#include "includes.h"
__global__ void hello(char *a, int *b)
{
for (int i=0; i<7; ++i)
{
a[i] += b[i];
}
}