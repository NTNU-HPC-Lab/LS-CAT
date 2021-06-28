#pragma once


namespace ip
{

// forward declaration
template <typename TPixel> class GpuImage;
template <typename TPixel> class CpuImage;

///
/// @brief Image class (CPU)
///
template <typename TPixel>
class CpuImage
{
    PixelType m_pixelType;
    int m_width;
    int m_height;
    int m_channels;
    int m_pixelSize;
    int m_depth;
    int m_pitch;

    roi_t m_roi;

    TPixel* m_pixelPointer;

public:

    CpuImage();
    CpuImage(int width, int height, int channels);
    CpuImage(const CpuImage<TPixel>& inst);
    explicit CpuImage(const GpuImage<TPixel>& inst);

    ~CpuImage();

    operator cpuImage_t<TPixel>() const;
    const CpuImage<TPixel>& operator=(const CpuImage<TPixel>& inst) = delete;

    PixelType PixelType() const;
    int Width() const;
    int Height() const;
    int Channels() const;
    int PixelSize() const;
    int Depth() const;
    int Pitch() const;
    roi_t Roi() const;
    roi_t& Roi();
    const TPixel* PixelPointer() const;
    TPixel* PixelPointer();

    void reconstruct(int width, int height, int channels);

private:
    void _allocate();
    void _deallocate();
};

typedef CpuImage<u8>  CpuImageU8;
typedef CpuImage<u16> CpuImageU16;
typedef CpuImage<u32> CpuImageU32;
typedef CpuImage<f32> CpuImageF32;
typedef CpuImage<f64> CpuImageF64;


} // namespace ip

