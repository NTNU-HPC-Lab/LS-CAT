#include "includes.h"

// This works fine with a mutex, but crashes with a sigbus error when not using a mutex
// #define USE_MUTEX

#ifdef USE_MUTEX
std::mutex m;
#endif


__global__ void testKernel() {
printf("Thread Kernel running\n");
}