#include "includes.h"
__global__ void from_2D_texture_to_memory_space(cudaTextureObject_t texture_source, float* destination, size_t w, size_t h) {

const uint2 gtid = {
threadIdx.x + blockIdx.x * blockDim.x,
threadIdx.y + blockIdx.y * blockDim.y
};
const auto gtid_serliazed = gtid.x + gtid.y * static_cast<unsigned>(w);

if (gtid.x < w && gtid.y < h) {
const float x = tex2D<float>(texture_source, gtid.x, gtid.y);
printf("Thread %u %u, reading value %4f, and writing to index %3u\n", gtid.x, gtid.y, x, gtid_serliazed);
destination[gtid_serliazed] = x;
}
}