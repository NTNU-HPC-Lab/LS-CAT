#include "includes.h"
__global__ void GoniometricFunctionKernel(float* input, float* output, const int size, const int type)
{
int id = blockDim.x * blockIdx.y * gridDim.x
+ blockDim.x * blockIdx.x
+ threadIdx.x;
if(id < size)
{	 // Sine = 0, Cosine = 1, Tan = 2, Tanh = 3, Sinh = 4, Cosh = 5  see MyGonioType in MyTransform.cs
switch (type)
{
case 0:
output[id] = sinf(input[id]);
break;
case 1:
output[id] = cosf(input[id]);
break;
case 2:
output[id] = tanf(input[id]);
break;
case 3:
output[id] = tanhf(input[id]);
break;
case 4:
output[id] = sinhf(input[id]);
break;
case 5:
output[id] = coshf(input[id]);
break;
case 6:
output[id] = asinf(input[id]);
break;
case 7:
output[id] = acosf(input[id]);
break;
case 10:
output[id] = atan2f(input[2*id], input[2*id+1]);
break;
}
}
}