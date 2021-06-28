#include "includes.h"
__device__ int addem( int a, int b ) {
return a + b;
}
__global__ void add( int a, int b, int *c ) {
*c = addem( a, b );
}