#pragma once

namespace proto
{

//template<typename TPixel>
//extern void calcHistogram(const ip::CpuImage<TPixel>& image, const unsigned int bins, unsigned int* hist);

extern void calcHistogram(const ip::CpuImageU8& image, const unsigned int bins, unsigned int* hist);
extern void calcHistogram(const ip::CpuImageU16& image, const unsigned int bins, unsigned int* hist);

//template<typename TPixel>
//extern void calcHistogram_AtomicOnly(const ip::GpuImage<TPixel>& image, const unsigned int bins, unsigned int* devHist);

extern void calcHistogram_AtomicOnly(const ip::GpuImageU8& image, const unsigned int bins, unsigned int* devHist);
extern void calcHistogram_AtomicOnly(const ip::GpuImageU16& image, const unsigned int bins, unsigned int* devHist);

}
