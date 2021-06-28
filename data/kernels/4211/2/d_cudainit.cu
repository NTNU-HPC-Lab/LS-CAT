#include "includes.h"

//#define ITEM_COUNT 2
#define _PI 3.14159265358979323846
#define _PI2 1.57079632679489661923
#define _RAD 6372795





using namespace std;

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);



__global__ void d_cudainit(int *a, int *b)
{
int i = threadIdx.x;
if (i==1)
{
b[i] = a[i] * 2;
}
}