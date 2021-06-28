#include "includes.h"


// This is my deviece function
// __global__ means this function is visible to the host

__global__ void kernelHelloWorld() {
printf("Hello World!\n");
}