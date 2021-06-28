#pragma once
#include "Exception.h"

#include <cstdint>

#include <cuda.h>
#include <nppdefs.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#define CHECK_CUDA_KERNEL() seba::check_cuda_kernel_launch()
#define CHECK_NPP_STATUS(status) seba::check_npp_status(status)

#define SEBA_VIDEO_CUDA_DEBUG 1

namespace seba
{
template <typename T>
thrust::device_ptr<T> device_malloc(size_t n)
{
	// Belt and suspenders as thrust documentation
	// isn't telling us anythimg about error
	try
	{
		auto ptr = thrust::device_malloc<T>(n);
		if (ptr.get())
			return ptr;
	}
	catch (...)
	{
	}
	SEBA_EXCEPTION(SEBA_INSUFFICIENT_DEVICE_MEMORY);
}

extern void check_cuda_kernel_launch();
extern void check_npp_status(NppStatus status);

template <typename T>
__host__ __device__ inline T Clamp(T val, T low, T high)
{
	if (val < low)
		return low;
	if (val > high)
		return high;
	return val;
}

template <const int N, typename T>
__host__ __device__ inline T Clamp(T val)
{
	return Clamp(val, (T)0, (T)((1 << N) - 1));
}

template <typename T>
typename T::value_type *PTR(T &container)
{
	return thrust::raw_pointer_cast(container.data());
}

class CudaTexture
{
  public:
	CudaTexture(int16_t *devPtr, size_t width, size_t height, size_t pitchElem = 0);
	CudaTexture(uint8_t *devPtr, size_t width, size_t height, size_t pitchElem = 0);
	virtual ~CudaTexture();
	operator cudaTextureObject_t();

  private:
	void Init(void *devPtr, size_t width, size_t height);
	cudaTextureObject_t m_texObj;
	cudaResourceDesc m_resDesc;
	cudaTextureDesc m_texDesc;
};

namespace debug
{

#ifdef SEBA_VIDEO_CUDA_DEBUG
extern void Save12bitsRGB(const int16_t *rImg, const int16_t *gImg, const int16_t *bImg, size_t width, size_t height, const char *name);
extern void Save12bitsYCC(const int16_t *Y, const int16_t *Cr, const int16_t *Cb, size_t width, size_t height, const char *name);
extern void Save12bitsGray(const int16_t *img, size_t width, size_t height, const char *name);
extern void Save8bitsRGB(const uint8_t *rImg, const uint8_t *gImg, const uint8_t *bImg, size_t width, size_t height, const char *name);
extern void Save8bitsYCC(const uint8_t *Y, const uint8_t *Cr, const uint8_t *Cb, size_t width, size_t height, const char *name);
extern void Save8bitsGray(const uint8_t *img, size_t width, size_t height, const char *name);
#endif

class Entroper
{
  public:
	Entroper(
		uint8_t *bytestream,
		const uint8_t *dc_luminance_val_spec,
		const uint8_t *dc_luminance_bits_spec,
		const uint8_t *dc_chrominance_val_spec,
		const uint8_t *dc_chrominance_bits_spec,
		const uint8_t *ac_luminance_val_spec,
		const uint8_t *ac_luminance_bits_spec,
		const uint8_t *ac_chrominance_val_spec,
		const uint8_t *ac_chrominance_bits_spec);

	void encode(int16_t *block, int chan);

	void flush();

	size_t size() const;

  private:
	void emit_bits_s(unsigned int code, int size);
	void emit_byte_s(char val);

	uint32_t m_dc_luminance_huff_code[256];
	uint8_t m_dc_luminance_huff_size[256];
	uint32_t m_dc_chrominance_huff_code[256];
	uint8_t m_dc_chrominance_huff_size[256];
	uint32_t m_ac_luminance_huff_code[256];
	uint8_t m_ac_luminance_huff_size[256];
	uint32_t m_ac_chrominance_huff_code[256];
	uint8_t m_ac_chrominance_huff_size[256];

	uint8_t *m_next_output_byte; /* => next byte to write in buffer */
	size_t m_buffer_len;

	struct savable_state
	{

		savable_state()
			: put_buffer(0),
			  put_bits(0)
		{
			last_dc_val[0] = 0;
			last_dc_val[1] = 0;
			last_dc_val[2] = 0;
			last_dc_val[3] = 0;
		}
		int32_t put_buffer; /* current bit-accumulation buffer */
		int put_bits;		/* # of bits now in it */
		int last_dc_val[4]; /* last DC coef for each component */
	};

	savable_state m_cur; /* Current bit buffer & DC state */
};
} // namespace debug

extern void DeriveHuffmanTable(
	const uint8_t *bits,
	const uint8_t *val,
	bool isDC,
	uint32_t ehufco[256],
	uint8_t ehufsi[256]);
} // namespace seba
