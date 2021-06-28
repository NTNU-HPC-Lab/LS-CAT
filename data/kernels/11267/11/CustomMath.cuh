#ifndef CUSTOMMATH_CUH
#define CUSTOMMATH_CUH
#include <math.h>
#include "cuda_runtime.h"

//reference : http://www.codeproject.com/Articles/69941/Best-Square-Root-Method-Algorithm-Function-Precisi first eq.
__host__ __device__ float sqrt1(const float x);
//reference :https://en.wikipedia.org/wiki/Fast_inverse_square_root
__host__ __device__ float reciprocal(float x);

#endif