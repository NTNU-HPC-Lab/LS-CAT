#include "includes.h"
/*****************************************************************************/
// nvcc -O1 -o bpsw bpsw.cu -lrt -lm


// Assertion to check for errors
__global__ void kernel_jacobi(long* nArray, long* dArray, long len) {
int bx = blockIdx.x;      // ID thread
int tx = threadIdx.x;
int result, t;
long d, dAbs, sign, temp, n1, d1;
// Identify the row and column of the Pd element to work on
long memIndex = bx*TILE_WIDTH + tx;
if (memIndex < len)							//out of bounds checking - some threads will be doing nothing
{
result = 0;
dAbs = 5;
sign = 1;

while (result != -1)				//if result != -1, increment d and try again
{
n1 = nArray[memIndex];				//reinitialize n1 to n
d = dAbs*sign;
t = 1;
d1 = d;							//reinitialize d1 to d
d1 = d1 % n1;

while (d1 != 0)
{
while (d1 % 2 == 0)        //while d is even
{
d1 = d1 / 2;
if (n1 % 8 == 3 || n1 % 8 == 5) t = -t;
}
temp = d1;
d1 = n1;
n1 = temp;
if ((d1 % 4 == 3) && (n1 % 4 == 3)) t = -t;
d1 = d1 % n1;
}
if (n1 == 1) result = t;
else result = 0;
dAbs = dAbs + 2;
sign = sign * -1;
}
}
__syncthreads();
if (memIndex < len)
dArray[memIndex] = d;
__syncthreads();
}