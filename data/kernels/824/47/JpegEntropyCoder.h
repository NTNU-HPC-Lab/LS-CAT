#pragma once

#include "seba-video/sebavideo_sdk.h"

namespace seba
{
extern void InitializeHuffmanTables(
	const uint8_t *dc_luminance_val_spec,
	const uint8_t *dc_luminance_bits_spec,
	const uint8_t *dc_chrominance_val_spec,
	const uint8_t *dc_chrominance_bits_spec,
	const uint8_t *ac_luminance_val_spec,
	const uint8_t *ac_luminance_bits_spec,
	const uint8_t *ac_chrominance_val_spec,
	const uint8_t *ac_chrominance_bits_spec);
extern __global__ void rleDpcmHuff_gpu(const int16_t *dct, uint32_t *bitstream, uint8_t *huffmanBlockTail, uint32_t *huffmanBlockSize, int channel, sebaJpegFormat_t format, int ri);
extern __global__ void alignBitstream_gpu(uint8_t *bitstream, uint8_t *huffmanBlockTail, uint32_t *huffmanBlockSize, uint32_t *huffmanBlockOffset, int ri);
extern __global__ void concatBitstream_gpu(const uint32_t *src, const uint32_t *huffmanBlockSize, uint32_t *huffmanBlockOffset, uint8_t *dst, uint32_t *bytestreamSize);
} // namespace seba
