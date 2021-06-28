#include "includes.h"
__global__ void add(int *result, int *num1, int *num2){
*result = *num1 + *num2;
}