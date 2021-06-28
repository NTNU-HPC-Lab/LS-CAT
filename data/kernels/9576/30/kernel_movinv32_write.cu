#include "includes.h"
__global__ void kernel_movinv32_write(char* _ptr, char* end_ptr, unsigned int pattern, unsigned int lb, unsigned int sval, unsigned int offset)
{
unsigned int i;
unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);

if (ptr >= (unsigned int*) end_ptr) {
return;
}

unsigned int k = offset;
unsigned pat = pattern;
for (i = 0;i < BLOCKSIZE/sizeof(unsigned int); i++){
ptr[i] = pat;
k++;
if (k >= 32){
k=0;
pat = lb;
}else{
pat = pat << 1;
pat |= sval;
}
}

return;
}