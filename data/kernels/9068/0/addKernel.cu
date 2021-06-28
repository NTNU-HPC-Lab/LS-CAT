#include "includes.h"
//기본 코드


cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);


//host에서 호출가능하며 Device에서 실행되는 함수 커널함수 정의

//host에서만 호출가능하며 host에서 실행되는 호스트 함수 정의
__global__ void addKernel(int *c, const int *a, const int *b)
{
int i = threadIdx.x; // kernel을 실행할 각 thread에게는 thread ID가 주어지는데, kernel 함수 내에서 built-in variable인 ‘threadIdx’로 액세스
c[i] = a[i] + b[i];
printf("%d\n", i);
}