#pragma once

#include "seba-video/sebavideo_sdk.h"

namespace seba
{
	class DeviceSurfaceBuffer;

	class JpegEncoder
	{
	public:

		static JpegEncoder *Create(
			unsigned maxWidth,
			unsigned maxHeight,
			const DeviceSurfaceBuffer &srcBuffer
		);

		virtual ~JpegEncoder();

		virtual void Encode(
			unsigned quality,
			sebaJfifInfo_t &jfifInfo
		) = 0;

	protected:

		JpegEncoder(
			unsigned maxWidth,
			unsigned maxHeight,
			const DeviceSurfaceBuffer &srcBuffer
		);

		int m_quality;
		const DeviceSurfaceBuffer *m_srcBuffer;
	};
}
