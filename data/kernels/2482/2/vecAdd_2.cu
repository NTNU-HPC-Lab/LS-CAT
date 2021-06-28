#include "includes.h"
__global__ void vecAdd_2(double *a, double *b, double *c, int n)
{
int id = threadIdx.x;
int id_1;

for(int i = 0; i < n; i++)
{
id_1 = id * n + i;

c[id_1] = a[id_1] + b[id_1];

}
}