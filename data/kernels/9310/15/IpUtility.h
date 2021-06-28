#pragma once

#include "IpException.h"

namespace ip
{

#define NOT_USED(X)   (void)(X);

//#if defined(_DEBUG)
//#define CHECK_ARG(CONDITION)    assert(CONDITION)
//#else
#define CHECK_ARG(CONDITION) \
    { \
        if (!(CONDITION)) \
        { \
            throw ip::IpException(); \
        } \
    }
//#endif

#if defined(__DRIVER_TYPES_H__)

#define CHECK_CUDA_ERROR(ERR) ip::checkCudaError((ERR), __FILE__, __LINE__, __FUNCTION__)
extern void checkCudaError(cudaError_t err, const char* file, unsigned int line, const char* function);


struct CudaMemoryDeleter
{
    void operator()(void* devPtr) const
    {
        cudaFree(devPtr);
    }
};


#endif

} // namespace ip
