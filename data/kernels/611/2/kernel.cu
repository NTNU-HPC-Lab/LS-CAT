#include "includes.h"
__global__ void kernel(const unsigned char * src, unsigned char * dst, int level, const size_t width, const size_t height)
{
const size_t xIndex = blockIdx.x * blockDim.x + threadIdx.x;
const size_t yIndex = blockIdx.y * blockDim.y + threadIdx.y;

if (xIndex < width && yIndex < height)
{
size_t o = yIndex * width + xIndex;
if (level == 256)
{
*(dst + o) = 0;
}
else
{
*(dst + o) = (*(src + o) >= level) ? 255 : 0;
}
// Notice how the below version avoids having an 'if' statement.
// I wonder if this is truly correct - I'll have to test this
// carefully someday but it works correctly. I figured the
// subtraction should cause an underflow which the shift might
// propagate through the rest of the byte so as to cause 255.
// *(dst + o) = ~((*(src + o) - level - 1) >> 7);
}
}