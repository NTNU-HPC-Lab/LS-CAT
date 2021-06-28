#include "includes.h"
using namespace std;


__global__ void AddIntsCUDA(int *a, int *b)
{
a[0] += b[0];
}