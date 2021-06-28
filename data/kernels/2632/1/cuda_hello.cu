#include "includes.h"
__global__ void cuda_hello(){
printf("Hello World from GPU!\n");
}