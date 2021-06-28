#pragma once

#include <cstdint>
#include <curand_kernel.h>

struct RNGGMem
{
    curandStateMRG32k3a_t* state;
};