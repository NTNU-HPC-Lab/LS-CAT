#include "includes.h"
__global__ void kernel_test1_write(char* _ptr, char* end_ptr, unsigned int* err)
{
unsigned int i;
unsigned long* ptr = (unsigned long*) (_ptr + blockIdx.x*BLOCKSIZE);

if (ptr >= (unsigned long*) end_ptr) {
return;
}


for (i = 0;i < BLOCKSIZE/sizeof(unsigned long); i++){
ptr[i] =(unsigned long) & ptr[i];
}

return;
}