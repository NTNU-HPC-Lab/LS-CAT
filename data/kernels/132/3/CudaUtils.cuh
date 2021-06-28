// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include <cuda_runtime.h>
#include "cuda/ComputeCapabilities.cuh"
#include "cuda/CudaStaticAssert.cuh"

namespace facebook { namespace cuda {

/**
   Computes ceil(a / b)
*/
template <typename T>
__host__ __device__ __forceinline__ constexpr T ceil(T a, T b) {
  return (a + b - 1) / b;
}

/**
   Computes floor(a / b)
*/
template <typename T>
__host__ __device__ __forceinline__ constexpr T floor(T a, T b) {
  return (a - b + 1) / b;
}

/**
   Returns the current thread's warp ID
*/
__device__ __forceinline__ int getWarpId() {
  return (threadIdx.z * blockDim.y * blockDim.x +
          threadIdx.y * blockDim.x +
          threadIdx.x) / warpSize;
}

/**
   Returns the number of threads in the current block (linearized).
*/
__device__ __forceinline__ int getThreadsInBlock() {
  return (blockDim.x * blockDim.y * blockDim.z);
}

/**
   Returns the number of warps in the current block (linearized,
   rounded to whole warps).
*/
__device__ __forceinline__ int getWarpsInBlock() {
  return ceil(getThreadsInBlock(), WARP_SIZE);
}

/**
   Pointer comparison using the PTX intrinsic; min() doesn't work for
   T*.
*/
template <typename T>
__device__ __forceinline__ T ptrMin(T a, T b) {
  cuda_static_assert(sizeof(T) == 8);

  T ptr;
  asm("min.u64 %0, %1, %2;" : "=l"(ptr) : "l"(a), "l"(b));
  return ptr;
}

/**
  Pointer comparison using the PTX intrinsic; max() doesn't work for
  T*
*/
template <typename T>
__device__ __forceinline__ T ptrMax(T a, T b) {
  cuda_static_assert(sizeof(T) == 8);

  T ptr;
  asm("max.u64 %0, %1, %2;" : "=l"(ptr) : "l"(a), "l"(b));
  return ptr;
}

/**
   Return the current thread's lane in the warp
*/
__device__ __forceinline__ int getLaneId() {
  int laneId;
  asm("mov.s32 %0, %laneid;" : "=r"(laneId) );
  return laneId;
}

/**
   Return a bitmask with bits set in positions less than the current
   thread's lane number in the warp.
*/
__device__ __forceinline__ unsigned getLaneMaskLt() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
  return mask;
}

/**
   Return a bitmask with bits set in positions less than or equal to
   the current thread's lane number in the warp.
*/
__device__ __forceinline__ unsigned getLaneMaskLe() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_le;" : "=r"(mask));
  return mask;
}

/**
   Return a bitmask with bits set in positions greater than the
   current thread's lane number in the warp.
*/
__device__ __forceinline__ unsigned getLaneMaskGt() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_gt;" : "=r"(mask));
  return mask;
}

/**
   Return a bitmask with bits set in positions greater than or equal
   to the current thread's lane number in the warp.
*/
__device__ __forceinline__ unsigned getLaneMaskGe() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_ge;" : "=r"(mask));
  return mask;
}

/**
   Extract a single bit at `pos` from `val`
*/
__device__ __forceinline__ int getBit(int val, int pos) {
  int ret;
  asm("bfe.u32 %0, %1, %2, 1;" : "=r"(ret) : "r"(val), "r"(pos));
  return ret;
}

/**
   Insert a single bit into `val` at position `pos`
*/
__device__ __forceinline__
unsigned setBit(unsigned val, unsigned toInsert, int pos) {
  unsigned ret;
  asm("bfi.b32 %0, %1, %2, %3, 1;" :
      "=r"(ret) : "r"(toInsert), "r"(val), "r"(pos));
  return ret;
}

/**
   Extract a bit field of length `len` at `pos` from `val`
*/
__device__ __forceinline__
unsigned getBitfield(unsigned val, int pos, int len) {
  unsigned ret;
  asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(val), "r"(pos), "r"(len));
  return ret;
}

/**
   Extract a bit field of length `len` at `pos` from `val`
*/
__device__ __forceinline__
unsigned long getBitfield(unsigned long val, int pos, int len) {
  unsigned long ret;
  asm("bfe.u64 %0, %1, %2, %3;" : "=l"(ret) : "l"(val), "r"(pos), "r"(len));
  return ret;
}

/**
   Insert `len` bits of `toInsert` into `val` starting at position
   `pos`
*/
__device__ __forceinline__
unsigned setBitfield(unsigned val, unsigned toInsert, int pos, int len) {
  unsigned ret;
  asm("bfi.b32 %0, %1, %2, %3, %4;" :
      "=r"(ret) : "r"(toInsert), "r"(val), "r"(pos), "r"(len));
  return ret;
}

/**
   Insert `len` bits of `toInsert` into `val` starting at position
   `pos`
*/
__device__ __forceinline__
unsigned long setBitfield(unsigned long val, unsigned long toInsert,
                          int pos, int len) {
  unsigned long ret;
  asm("bfi.b64 %0, %1, %2, %3, %4;" :
      "=l"(ret) : "l"(toInsert), "l"(val), "r"(pos), "r"(len));
  return ret;
}

/**
   Returns the index of the most significant 1 bit in `val`.
*/

__device__ __forceinline__ constexpr int getMSB(int val) {
  return
    ((val >= 1024 && val < 2048) ? 10 :
     ((val >= 512) ? 9 :
      ((val >= 256) ? 8 :
       ((val >= 128) ? 7 :
        ((val >= 64) ? 6 :
         ((val >= 32) ? 5 :
          ((val >= 16) ? 4 :
           ((val >= 8) ? 3 :
            ((val >= 4) ? 2 :
             ((val >= 2) ? 1 :
              ((val == 1) ? 0 : -1)))))))))));
}

} }  // namespace
