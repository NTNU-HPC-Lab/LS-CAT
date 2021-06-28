#include "includes.h"
//-----include header files, ¤Þ¤J¼ÐÀYÀÉ-----


cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);


__global__ void addKernel(int *c, const int *a, const int *b)			//	addKernel¨ç¼Æ
{																		//	addKernel function, addKernel¨ç¼Æ
int i = threadIdx.x;
c[i] = a[i] + b[i];
}