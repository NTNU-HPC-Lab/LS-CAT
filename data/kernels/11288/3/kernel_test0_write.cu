#include "includes.h"
__global__ void kernel_test0_write(char* _ptr, char* end_ptr, unsigned int pattern, unsigned int* err, unsigned long* err_addr, unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read)
{
unsigned int i;
unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);

if (ptr >= (unsigned int*) end_ptr) {
return;
}


for (i = 0;i < BLOCKSIZE/sizeof(unsigned int); i++){
ptr[i] = pattern;
}

return;
}