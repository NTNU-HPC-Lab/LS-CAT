#pragma once

#include <cuda_runtime.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600

__device__ __inline__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address64 = (unsigned long long int *)address;
    unsigned long long int old = *address64;
    unsigned long long int assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address64, assumed, 
            __double_as_longlong(__longlong_as_double(assumed) + val));

    } while (old != assumed);

    return __longlong_as_double(old);
}

#endif 

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 200

__device__ __inline__ float atomicAdd(float* address, float val)
{
    int* address32 = (int*)address;
    int old = *address32;
    int assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address32, assumed,
            __float_as_int(__int_as_float(assumed) + val));

    } while (old != assumed);

    return __int_as_float(old);
}

#endif


__device__ __inline__ long long int atomicAdd(long long int* address, long long int val)
{
    unsigned long long int* addressU64 = (unsigned long long int*)address;
    unsigned long long int old = *addressU64;
    unsigned long long int assumed;

    do
    {
        assumed = old;
        old = atomicCAS(addressU64, assumed,
            (unsigned long long int)((long long int)assumed + val));

    } while (old != assumed);

    return (long long int)old;
}

__device__ __inline__ long long int atomicSub(long long int* address, long long int val)
{
    unsigned long long int* addressU64 = (unsigned long long int*)address;
    unsigned long long int old = *addressU64;
    unsigned long long int assumed;

    do
    {
        assumed = old;
        old = atomicCAS(addressU64, assumed,
            (unsigned long long int)((long long int)assumed - val));

    } while (old != assumed);

    return (long long int)old;
}

__device__ __inline__ unsigned long long int atomicSub(unsigned long long int* address, unsigned long long int val)
{
    unsigned long long int old = *address;
    unsigned long long int assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address, assumed, assumed - val);

    } while (old != assumed);

    return old;
}

__device__ __inline__ float atomicSub(float* address, float val)
{
    int* address32 = (int*)address;
    int old = *address;
    int assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address32, assumed,
            __float_as_int(__int_as_float(assumed) - val));

    } while (old != assumed);

    return __int_as_float(old);
}

__device__ __inline__ double atomicSub(double* address, double val)
{
    unsigned long long int* address64 = (unsigned long long int*)address;
    unsigned long long int old = *address;
    unsigned long long int assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address64, assumed,
            __double_as_longlong(__longlong_as_double(assumed) - val));

    } while (old != assumed);

    return __longlong_as_double(old);
}

__device__ __inline__ long long int atomicExch(long long int* address, long long int val)
{
    return (long long int)atomicExch((unsigned long long int*)address, (unsigned long long int)val);
}

__device__ __inline__ double atomicExch(double* address, double val)
{
    unsigned long long int* address64 = (unsigned long long int*)address;
    unsigned long long int old = *address64;
    unsigned long long int assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address64, assumed, __double_as_longlong(val));

    } while (old != assumed);

    return __longlong_as_double(old);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 350

__device__ __inline__ long long int atomicMin(long long int* address, long long int val)
{
    unsigned long long int* addressU64 = (unsigned long long int*)address;
    unsigned long long int old = *addressU64;
    unsigned long long int assumed;

    do
    {
        assumed = old;
        old = atomicCAS(addressU64, assumed, min((long long int)assumed, val));

    } while (old != assumed);

    return (long long int)old;
}

#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 350

__device__ __inline__ unsigned long long int atomicMin(unsigned long long int* address, unsigned long long int val)
{
    unsigned long long int old = *address;
    unsigned long long int assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address, assumed, min(assumed, val));

    } while (old != assumed);

    return old;
}

#endif


__device__ __inline__ float atomicMin(float* address, float val)
{
    int* address32 = (int*)address;
    int old = *address32;
    int assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address32, assumed, 
            __float_as_int(min(__int_as_float(assumed), val)));

    } while (old != assumed);

    return __int_as_float(old);
}

__device__ __inline__ double atomicMin(double* address, double val)
{
    unsigned long long int* address64 = (unsigned long long int*)address;
    unsigned long long int old = *address64;
    unsigned long long int assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address64, assumed,
            __double_as_longlong(min(__longlong_as_double(assumed), val)));

    } while (old != assumed);

    return __longlong_as_double(old);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 350

__device__ __inline__ long long int atomicMax(long long int* address, long long int val)
{
    unsigned long long int* addressU64 = (unsigned long long int*)address;
    unsigned long long int old = *addressU64;
    unsigned long long int assumed;

    do
    {
        assumed = old;
        old = atomicCAS(addressU64, assumed, max((long long int)assumed, val));

    } while (old != assumed);

    return (long long int)old;
}

#endif


#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 350

__device__ __inline__ unsigned long long int atomicMax(unsigned long long int* address, unsigned long long int val)
{
    unsigned long long int old = *address;
    unsigned long long int assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address, assumed, max(assumed, val));

    } while (old != assumed);

    return old;
}

#endif

__device__ __inline__ float atomicMax(float* address, float val)
{
    int* address32 = (int*)address;
    int old = *address32;
    int assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address32, assumed,
            __float_as_int(max(__int_as_float(assumed), val)));

    } while (old != assumed);

    return old;
}

__device__ __inline__ double atomicMax(double* address, double val)
{
    unsigned long long int* address64 = (unsigned long long int*)address;
    unsigned long long int old = *address64;
    unsigned long long int assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address64, assumed,
            __double_as_longlong(max(__longlong_as_double(assumed), val)));

    } while (old != assumed);

    return old;
}

__device__ __inline__ long long int atomicAnd(long long int* address, long long int val)
{
    unsigned long long int* address64 = (unsigned long long int*)address;
    unsigned long long int old = *address64;
    unsigned long long int assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address64, assumed, ((long long int)assumed) & val);

    } while (old != assumed);

    return (long long int)old;
}


#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 350

__device__ __inline__ unsigned long long int atomicAnd(unsigned long long int* address, unsigned long long int val)
{
    unsigned long long int old = *address;
    unsigned long long int assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address, assumed, assumed & val);

    } while (old != assumed);

    return old;
}

#endif


__device__ __inline__ long long int atomicOr(long long int* address, long long int val)
{
    unsigned long long int* address64 = (unsigned long long int*)address;
    unsigned long long int old = *address64;
    unsigned long long int assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address64, assumed, ((long long int)assumed) | val);

    } while (old != assumed);

    return (long long int)old;
}


#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 350

__device__ __inline__ unsigned long long int atomicOr(unsigned long long int* address, unsigned long long int val)
{
    unsigned long long int old = *address;
    unsigned long long int assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address, assumed, assumed | val);

    } while (old != assumed);

    return old;
}

#endif

__device__ __inline__ long long int atomicXor(long long int* address, long long int val)
{
    unsigned long long int* address64 = (unsigned long long int*)address;
    unsigned long long int old = *address64;
    unsigned long long int assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address64, assumed, ((long long int)assumed) ^ val);

    } while (old != assumed);

    return (long long int)old;
}


#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 350

__device__ __inline__ unsigned long long int atomicXor(unsigned long long int* address, unsigned long long int val)
{
    unsigned long long int old = *address;
    unsigned long long int assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address, assumed, assumed ^ val);

    } while (old != assumed);

    return old;
}

#endif


