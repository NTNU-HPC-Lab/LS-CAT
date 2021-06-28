#pragma once
/**
*/

#include <map>
#include "DeviceMemory.h"
#include "RNGStructs.h"

class CudaGPU;
class CudaSystem;

class RNGMemory
{
    private:
        DeviceMemory                        memRandom;
        std::map<const CudaGPU*, RNGGMem>   randomStacks;

    protected:
    public:
        // Constructors & Destructor
                                            RNGMemory() = default;
                                            RNGMemory(uint32_t seed,
                                                      const CudaSystem&);
                                            RNGMemory(const RNGMemory&) = delete;
                                            RNGMemory(RNGMemory&&) = default;
        RNGMemory&                          operator=(const RNGMemory&) = delete;
        RNGMemory&                          operator=(RNGMemory&&) = default;
                                            ~RNGMemory() = default;

        RNGGMem                             RNGData(const CudaGPU&);
};