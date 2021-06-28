#include "includes.h"

// CUDA kernel. Each thread takes care of one element of c

__global__ void encode(char *encodedText, char *decodedText)
{
// Get our global thread ID
int id = blockIdx.x*blockDim.x+threadIdx.x;
int startEncoded = id * 101;
int startDecoded = id * 4;
int t,finish=startEncoded+100;
// Make sure we do not go out of bounds
if (id < 15360)
{
for(t=startEncoded;t<finish;t++)
{
if(encodedText[t]==',')
{
decodedText[startDecoded]=encodedText[t+1];
startDecoded++;
}
}
}
}