#include "includes.h"




__global__ void vecadd( int * v0, int * v1, std::size_t size )
{
auto tid = threadIdx.x;
v0[ tid ] += v1[ tid ];
}