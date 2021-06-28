#include "includes.h"





using namespace std;



__global__ void GPU_mt_info()
{
printf("Block idx: %d | thread idx: %d\n", blockIdx.x, threadIdx.x);
}