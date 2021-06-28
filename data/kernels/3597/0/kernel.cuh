#include "cuda_runtime.h"
#include <cstdint>

cudaError_t addWithCuda(int32_t *c, const int32_t *a, const int32_t *b, uint32_t size);

