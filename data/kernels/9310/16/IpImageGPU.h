#pragma once


namespace ip
{

// forward declaration
template <typename TPixel> class CpuImage;
template <typename TPixel> class GpuImage;


///
/// @brief Image class (GPU)
///
template <typename TPixel>
class GpuImage
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

    GpuImage();
    GpuImage(int width, int height, int channels);
    GpuImage(const GpuImage<TPixel>& inst);
    explicit GpuImage(const CpuImage<TPixel>& inst);

    ~GpuImage();

    operator gpuImage_t<TPixel>() const;
    const GpuImage<TPixel>& operator=(const GpuImage<TPixel>& inst) = delete;

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


typedef GpuImage<u8>  GpuImageU8;
typedef GpuImage<u16> GpuImageU16;
typedef GpuImage<u32> GpuImageU32;
typedef GpuImage<f32> GpuImageF32;
typedef GpuImage<f64> GpuImageF64;

} // namespace ip

