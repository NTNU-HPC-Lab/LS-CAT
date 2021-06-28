#ifndef _ped_region_h_
#define _ped_region_h_

#include <utility>

namespace Ped {
	class Region {
	public:
		int isInside(float x, float y);

		std::pair<int, int> getCenter();

		float getInnerRadius();

		float getOuterRadius();

		Region(std::pair<int, int> center, float innerRadius, float outerRadius);

	private:
		std::pair<int, int> center;

		float innerRadius;

		float outerRadius;
	};
}

#endif
