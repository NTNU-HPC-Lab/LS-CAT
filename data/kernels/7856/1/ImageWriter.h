// ImageWriter.h -- Abstract Base Class for writing image files

#ifndef IMAGEWRITER_H
#define IMAGEWRITER_H

#include <string>

class ImageWriter
{
public:
	virtual ~ImageWriter();

	virtual void	closeImageFile() = 0;

	// PRIMARY WRITING INTERFACE.
	// Choose EITHER (1) write scan line by scan line OR (2) write
	// entire file from a complete internal "frame buffer" array.
	// Results are undefined if you try to mix line-by-line writing
	// with "write all at once"!

	// (1) Image files can be written scan line by scan line using
	//     addScanLine. (It is permissable to mix these two.)
	//     'sLine' is assumed to be xRes*numChannels elements long
	//     with all the channels for one pixel stored consecutively.
	virtual void	addScanLine(const double* sLine) = 0;
	virtual void	addScanLine(const unsigned char* sLine) = 0;
	// OR
	// (2) Image files can be written all at once using writeImage.
	//     'fb' is assumed to be stored in row-major order and
	//     indexed as fb[row][col][channel].
	virtual void	writeImage(const unsigned char* fb) = 0;
	// END: PRIMARY WRITING INTERFACE.

	// Other methods (NOTE: Height <-> yRes; Width <-> xRes)
	int getHeight() const { return mYRes; }
	int getNumChannels() const { return mNumChannels; }
	int getWidth() const { return mXRes; }

	// A factory method that will create an ImageWriter based on the file name suffix
	//                                                 WIDTH    HEIGHT
	static ImageWriter* create(std::string fileName, int xres, int yres, int numChannels=3);

protected:
	ImageWriter(std::string fName, int xres, int yres, int numChannels=3);
	ImageWriter(const ImageWriter& iw); // disallow copy constructor

	static ImageWriter* guessFileType(const std::string& fileName, int xres, int yres, int numChannels);

	std::string			mImageFileName;
	int					mXRes; // width
	int					mYRes; // height
	int					mNumChannels;
};

#endif
