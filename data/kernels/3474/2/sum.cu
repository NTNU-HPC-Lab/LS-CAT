#include "includes.h"
__global__ void sum(int *a, int *b, int *c){
*c = *a + *b;
}