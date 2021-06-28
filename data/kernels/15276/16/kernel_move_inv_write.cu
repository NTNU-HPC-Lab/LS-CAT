#include "includes.h"
__global__ void kernel_move_inv_write(char* _ptr, char* end_ptr, unsigned int pattern)
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