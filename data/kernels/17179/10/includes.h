#define KERNEL_POSITION											\
int position = (blockDim.x * blockIdx.x + threadIdx.x);		\
//new series 
