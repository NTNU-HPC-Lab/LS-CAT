#include "includes.h"
/**
* Vector Addition - Simple addition using Cuda.
* Author - Malhar Bhatt
* Subject - High Performance Computing
*/


/** Function Add -
* Usage - Add 2 values
* Returns - Void
*/
__global__ void add( int num1, int num2, int *ans )
{
*ans = num1 + num2;
}