#include "includes.h"

using namespace std;



__global__ void addition(int *a, int *b, int *c)
{
*c = *a + *b;
}