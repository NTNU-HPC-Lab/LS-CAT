#pragma once

#include "seba-video/sebavideo_sdk.h"
#include "JpegEncoder.h"

#include <cstdint>

#include <thrust/device_vector.h>

namespace seba
{
class DeviceSurfaceBuffer;

class J12Encoder : public JpegEncoder
{
  public:
	J12Encoder(
		unsigned maxWidth,
		unsigned maxHeight,
		const DeviceSurfaceBuffer &srcBuffer);

	virtual ~J12Encoder();

	virtual void Encode(
		unsigned quality,
		sebaJfifInfo_t &jfifInfo);

  protected:
	void Do444Encode(
		int16_t *yImg,
		int16_t *crImg,
		int16_t *cbImg,
		sebaJfifInfo_t &jfifInfo);

	void Do422Encode(
		int16_t *yImg,
		int16_t *crImg,
		int16_t *cbImg,
		sebaJfifInfo_t &jfifInfo);

	void Do420Encode(
		int16_t *yImg,
		int16_t *crImg,
		int16_t *cbImg,
		sebaJfifInfo_t &jfifInfo);

	void DoYEncode(
		int16_t *yImg,
		sebaJfifInfo_t &jfifInfo);

	struct CudaGridConfig
	{
		dim3 block, grid;
	} m_colorSpaceConvertionGrid,
		m_422DownSamplingGrid,
		m_420DownSamplingGrid;

	void ComputeGridConfigs(size_t width, size_t height);

	void InitializeQTables(unsigned quality, sebaJfifInfo_t &jfifInfo);

	void AllocateWorkspace(unsigned maxWidth, unsigned maxHeight);

	thrust::device_vector<int16_t> m_downSampledCrImg;
	thrust::device_vector<int16_t> m_downSampledCbImg;
	thrust::device_vector<int16_t> m_yDCT;
	thrust::device_vector<int16_t> m_crDCT;
	thrust::device_vector<int16_t> m_cbDCT;
	thrust::device_vector<uint8_t> m_deviceScanStorage;
	thrust::device_vector<uint32_t> m_huffmanBlockSize;
	thrust::device_vector<uint32_t> m_huffmanBlockOffset;
	thrust::device_vector<uint8_t> m_huffmanBlockTail;
	thrust::device_vector<uint32_t> m_bitstream;
	thrust::device_vector<uint32_t> m_bytestreamSize;
};
} // namespace seba
