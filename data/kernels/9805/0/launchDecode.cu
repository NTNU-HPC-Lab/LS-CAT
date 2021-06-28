#include "includes.h"
__device__ void testPrint(){
printf("DEVICE PRINT \n");
}
__global__ void launchDecode() {
printf("RUNNING\n");
testPrint();
}