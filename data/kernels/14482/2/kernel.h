#ifndef KERNEL_H
#define KERNEL_H

__global__ void kernel_add(char* newB, char* first, char* second, int size_biggest, int diff, int * size_newB);
__global__ void kernel_sub(char* newB, char* first, char* second, int size_biggest, int diff, int * size_newB);
__global__ void kernel_mul(char* newB, char* first, char* second, int size_first, int size_second, int * size_newB);
__global__ void kernel_div(char* newB, char* first, char* second, int size_first, int size_second, int * size_newB, char* aux);

__global__ void kernel_fact(char* newB, const char* first, int size_first, int * size_newB);
__global__ void kernel_GCD(char* newB, const char* first, int size_first, int * size_newB);

#endif
