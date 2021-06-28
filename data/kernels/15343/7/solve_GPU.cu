#include "includes.h"
__global__ void solve_GPU(int a, int b, int c ,int *x1, int *x2)
{
int raiz = powf(b, 2) - (4 * a * c);
int i = -b / 2 * a;
int j = 2 * a;

*x1 = i + sqrtf(raiz) / j;
*x2 = i - sqrtf(raiz) / j;
}