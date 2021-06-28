#include "includes.h"

using namespace std;




__global__ void AddIntsCuda(int *a, int *b)
{
int i = threadIdx.x;
a[i] += b[i];
}