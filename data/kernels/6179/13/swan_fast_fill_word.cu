#include "includes.h"
__global__ void swan_fast_fill_word( uint *ptr, int len ) {
int idx = threadIdx.x + blockDim.x * blockIdx.x;
if( idx<len) {
ptr[idx] = 0;
}
}