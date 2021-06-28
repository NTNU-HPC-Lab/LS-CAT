#include "includes.h"
/*
* Implements vector
*/

#ifdef DEBUG
#endif



__global__ void kern_vec_add_(float* x, float* y, float* r, size_t dim)
{
size_t _strd = blockDim.x * gridDim.x;
for(size_t _i = blockIdx.x * blockDim.x + threadIdx.x; _i < dim; _i += _strd)
r[_i] = x[_i] + y[_i];
}