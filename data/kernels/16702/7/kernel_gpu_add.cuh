#pragma once

#include "cpu_to_gpu_mem.h"

#ifdef __cplusplus									// nvcc in C compiler olarak compile etmesi icin C dosyasi oldugunu belirt
extern "C"
#endif // __cplusplus

void cpu_gpu_execute(struct cpu_gpu_mem* cgm);		// __global__ prefixli yani GPU fonksiyonunu kullanan fonksiyonlarin prototopi burada yapilmalidir