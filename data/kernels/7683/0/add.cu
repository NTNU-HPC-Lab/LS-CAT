#include "includes.h"

using namespace std;

__global__ void add(int a, int b, int *c)//kernel函数，在gpu上运行。
{
*c = a + b;
}