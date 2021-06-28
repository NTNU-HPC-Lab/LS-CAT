//
// Created by psaw on 9/17/16.
//

#include "cuda_smpp_util.h"

__device__ void decodeSinglePdu(CudaPduContext *pduContext, CudaDecodedContext *decodedPduStruct, uint8_t *pduBuffer);
