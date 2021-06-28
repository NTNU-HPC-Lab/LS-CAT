#pragma once

namespace ip
{

extern void calcHistogram(const ip::CpuImageU8&  image, unsigned int bins, unsigned int* hist);
extern void calcHistogram(const ip::CpuImageU16& image, unsigned int bins, unsigned int* hist);

extern void calcHistogram(const ip::GpuImageU8&  image, unsigned int bins, unsigned int* devHist);
extern void calcHistogram(const ip::GpuImageU16& image, unsigned int bins, unsigned int* devHist);

} // namespace ip

