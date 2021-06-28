#include "includes.h"
__global__  void kernelSoftmax( float* x, int channels, float* y)
{

extern __shared__ float mem[];
__shared__ float sum_value;

float number = *(x + blockDim.x*blockIdx.x + threadIdx.x);
float number_exp = __expf(number);

//    sum_value += number_exp ;
/* *
* @TODO: Can do with the help of atomicAdd.
* */
atomicAdd(&sum_value, number_exp);
__syncthreads();

//	mem[threadIdx.x] = number_exp;

/* *
* @TODO: Can do with the help of a for loop. Try different methods and find the time taken.
* */
//	float sum = 0.0f;
//	for (int i=0;i<channels;i++)
//	{
//		sum += mem[i];
//	}

y[blockDim.x*blockIdx.x + threadIdx.x] = __fdiv_rd(number_exp, sum_value);

}