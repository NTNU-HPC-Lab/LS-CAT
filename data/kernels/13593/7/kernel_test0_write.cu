#include "includes.h"
__global__ void kernel_test0_write(char* _ptr, char* end_ptr)
{

unsigned int* orig_ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);;
unsigned int* ptr = orig_ptr;
if (ptr >= (unsigned int*) end_ptr) {
return;
}

unsigned int* block_end = orig_ptr + BLOCKSIZE/sizeof(unsigned int);

unsigned int pattern = 1;

unsigned long mask = 4;

*ptr = pattern;

while(ptr < block_end){

ptr = (unsigned int*) ( ((unsigned long)orig_ptr) | mask);
if (ptr == orig_ptr){
mask = mask <<1;
continue;
}
if (ptr >= block_end){
break;
}

*ptr = pattern;

pattern = pattern << 1;
mask = mask << 1;
}
return;
}