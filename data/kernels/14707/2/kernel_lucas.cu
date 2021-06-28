#include "includes.h"
/*****************************************************************************/
// nvcc -O1 -o bpsw bpsw.cu -lrt -lm


// Assertion to check for errors
__global__ void kernel_lucas(long* nArray, long* dArray, int* rArray, long len) {
int bx = blockIdx.x;      // ID thread
int tx = threadIdx.x;
int i, length;
long long d, n;
long long q, q2, u, u2, uold, v, v2, t;

// Identify the row and column of the Pd element to work on
long memIndex = bx*TILE_WIDTH + tx;
if (memIndex < len)							//out of bounds checking - some threads will be doing nothing
{
d = (long long) dArray[memIndex];
n = (long long) nArray[memIndex];
q = (1 - d) / 4;
u = 0;
v = 2;
u2 = 1;
v2 = 1;
q2 = 2 * q;
t = (n + 1) / 2;						//theta
length = 32 - __clz(t); //length of our number in bits. //clz(b00010010) = 3

for (i = 0; i < length; i++)
{
u2 = (u2 * v2) % n;
v2 = (v2 * v2 - q2) % n;
if (t & 1)				//mask = 1
{
uold = u;
u = (u2 * v) + (u * v2);
u = (u % 2 == 1) ? u + n : u;
u = (u / 2) % n;
v = (v2 * v) + (u2 * uold * d);
v = (v % 2 == 1) ? v + n : v;
v = (v / 2) % n;
}

q = (q*q) % n;
q2 = q + q;

t = t >> 1;
}

}
__syncthreads();
if (memIndex < len)
rArray[memIndex] = (u == 0);

}