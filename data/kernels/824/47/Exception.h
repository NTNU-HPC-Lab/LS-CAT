#pragma once

#include <seba-video/sebavideo_sdk.h>

#define SEBA_EXCEPTION(code) throw seba::Exception(code)

namespace seba
{
class Exception
{
public:
	Exception(sebaStatus_t status)
			: m_status(status)
	{
	}

	virtual ~Exception()
	{
	}

	sebaStatus_t GetStatus() const
	{
		return m_status;
	}

protected:
	sebaStatus_t m_status;
};
} // namespace seba
