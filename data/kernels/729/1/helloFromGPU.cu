#include "includes.h"
__global__ void helloFromGPU(){
if(threadIdx.x == 5){
printf("Hello World form GPU! thread %d\n",threadIdx.x);
}
}