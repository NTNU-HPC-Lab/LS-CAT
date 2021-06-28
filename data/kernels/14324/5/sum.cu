#include "includes.h"
__global__ void sum(int *a, int *b, int *c)
{
int i;
for(i = 0; i < N; i++) {
c[i] = a[i] + b[i];
}
}