#include "includes.h"
__global__ void kernel_modtest_write(char* _ptr, char* end_ptr, unsigned int offset, unsigned int p1, unsigned int p2)
{
unsigned int i;
unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);

if (ptr >= (unsigned int*) end_ptr) {
return;
}

for (i = offset;i < BLOCKSIZE/sizeof(unsigned int); i+=MOD_SZ){
ptr[i] =p1;
}

for (i = 0;i < BLOCKSIZE/sizeof(unsigned int); i++){
if (i % MOD_SZ != offset){
ptr[i] =p2;
}
}

return;
}