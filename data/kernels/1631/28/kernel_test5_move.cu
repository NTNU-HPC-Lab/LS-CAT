#include "includes.h"
__global__ void kernel_test5_move(char* _ptr, char* end_ptr)
{
unsigned int i;
unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);

if (ptr >= (unsigned int*) end_ptr) {
return;
}

unsigned int half_count = BLOCKSIZE/sizeof(unsigned int)/2;
unsigned int* ptr_mid = ptr + half_count;

for (i = 0;i < half_count; i++){
ptr_mid[i] = ptr[i];
}

for (i=0;i < half_count - 8; i++){
ptr[i + 8] = ptr_mid[i];
}

for (i=0;i < 8; i++){
ptr[i] = ptr_mid[half_count - 8 + i];
}

return;
}