#pragma once

#include "seba-video/sebavideo_sdk.h"

#include <cstdint>

#include <thrust/device_ptr.h>

namespace seba
{
int SurfacePixelByteSize(sebaSurfaceFormat_t surfaceFmt);
int SurfacePixelBitSize(sebaSurfaceFormat_t surfaceFmt);

enum StorageFlavor
{
	Planar,
	Packed
};

class DeviceSurfaceBuffer
{
  public:
	DeviceSurfaceBuffer(
		sebaSurfaceFormat_t surfaceFmt,
		StorageFlavor storageFlavor,
		unsigned maxWidth,
		unsigned maxHeight);

	virtual ~DeviceSurfaceBuffer();

	sebaDeviceSurfaceBufferInfo_t &GetInfo();
	const sebaDeviceSurfaceBufferInfo_t &GetInfo() const;

	// That should could go in sebaDeviceSurfaceBufferInfo_t
	// but I don't want to break compatibility with fast video.
	StorageFlavor GetStorageFlavor() const;
	void SetStorageFlavor(StorageFlavor flavor);

	uint8_t *GetDevicePtr();
	const uint8_t *GetDevicePtr() const;

  protected:
	sebaDeviceSurfaceBufferInfo_t m_info;
	thrust::device_ptr<uint8_t> m_buffer;
	StorageFlavor m_storageFlavor;
};
} // namespace seba
