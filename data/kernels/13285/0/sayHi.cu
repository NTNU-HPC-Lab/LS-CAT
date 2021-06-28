#include "includes.h"


using namespace std;


#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)
__global__ void sayHi()
{
printf("Cuda Kernel Hello Word.\n");
}