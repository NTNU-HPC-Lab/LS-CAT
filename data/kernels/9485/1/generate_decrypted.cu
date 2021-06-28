#include "includes.h"
__global__ void generate_decrypted(int *pDataPointer , int *pRandomData , int *pEncryptedData , long long int pSize)
{
long long int index = blockIdx.x * blockDim.x + threadIdx.x;
if( index <=(pSize /sizeof(int) ))
{
(*(pEncryptedData+index)) = (*(pDataPointer+ index))^(*(pRandomData+index));
}
else
return;
}