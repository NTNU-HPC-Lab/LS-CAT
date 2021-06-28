#include "includes.h"
__global__ void unsafe(int *shared_var, int iters)
{
for (int i = 0; i < iters; i++)
{
int old = *shared_var;
*shared_var = old + 1;
}
}