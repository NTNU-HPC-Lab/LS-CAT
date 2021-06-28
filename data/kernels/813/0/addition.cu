#include "includes.h"
/* Addition of two numbers using a kernel method.
* Note: Documentation will explain each thing only once. */

/* Header files */


/* This a kernel function, it has the __global__ qualifier in the definition.
* addition: Perform the addition of two numbers and return their sum.
* +------------+-----------------------------------+
* | Parameters | Description                       |
* +------------+-----------------------------------+
* | int  a     | Takes an integer passed by value. |
* | int  b     | Takes an integer passed by value. |
* | int *c     | An integer pointer that refers to |
* |            | the GPU memory where we store the |
* |            | result of the addition.           |
* +------------+-----------------------------------+ */


/* main method runs on the host. */
__global__ void addition ( int a, int b, int *c )
{
*c = a + b;
}