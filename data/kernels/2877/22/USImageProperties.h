// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2016, all rights reserved,
//      Rüdiger Göbl 
//		Email r.goebl@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
// 
// ================================================================================================

#ifndef __USIMAGEPROPERTIES_H__
#define __USIMAGEPROPERTIES_H__

#include <memory>
#include <vector>
#include <map>
#include "vec.h"

#include <iostream>
#include <iomanip>

#include <utilities/utility.h>

namespace supra
{
	// The receive parameters for one scanline
	struct ScanlineRxParameters3D
	{
		ScanlineRxParameters3D()
			: txParameters{ {{0,0}, {0,0}, 0, 0} }
			, position{ 0.0, 0.0, 0.0 }
			, direction{ 0.0, 0.0, 0.0 }
			, maxElementDistance{ 0.0, 0.0 }
		{}

		struct TransmitParameters {
			vec2T<uint16_t> firstActiveElementIndex;	// index of the first active transducer element
			vec2T<uint16_t> lastActiveElementIndex;		// index of the last active transducer element
			uint16_t  txScanlineIdx;		// index of the corresponsing transmit scanline
			double initialDelay;			// the minmal delay in [s] that is to be used during rx

			bool operator== (const TransmitParameters& b) const
			{
				return firstActiveElementIndex == b.firstActiveElementIndex &&
					lastActiveElementIndex == b.lastActiveElementIndex &&
					txScanlineIdx == b.txScanlineIdx &&
					initialDelay == b.initialDelay;
			}
		};

		vec position;	                // the position of the scanline
		vec direction;                  // direction of the scanline
		double txWeights[4];            // Weights for interpolation between different transmits
		TransmitParameters txParameters[4]; // Parameters of the transmits to use
		vec2 maxElementDistance;		// maximum distance of an element to the scanline start, used to compute rxWeights

		vec getPoint(double depth) const
		{
			return position + depth*direction;
		}

		bool operator== (const ScanlineRxParameters3D& b) const
		{
			return txParameters[0] == b.txParameters[0] &&
				txParameters[1] == b.txParameters[1] &&
				txParameters[2] == b.txParameters[2] &&
				txParameters[3] == b.txParameters[3] &&
				txWeights[0] == b.txWeights[0] &&
				txWeights[1] == b.txWeights[1] &&
				txWeights[2] == b.txWeights[2] &&
				txWeights[3] == b.txWeights[3] &&
				position == b.position &&
				direction == b.direction &&
				maxElementDistance == b.maxElementDistance;
		}

		friend std::ostream& operator<< (std::ostream& os, const ScanlineRxParameters3D& params);
		friend std::istream& operator>> (std::istream& is, ScanlineRxParameters3D& params);
	};


	class USImageProperties
	{
	public:
		enum ImageType {
			BMode,
			Doppler,
			Planewave
		};

		enum ImageState {
			Raw,
			RawDelayed,
			RF,
			EnvDetected,
			PreScan,
			Scan
		};
		enum TransducerType {
			Linear,
			Phased,
			Curved,
			Planar,
			PlanarPhased,
			Bicurved
		};

		USImageProperties();
		USImageProperties(vec2s scanlineLayout, size_t numSamples, USImageProperties::ImageType imageType, USImageProperties::ImageState imageState, USImageProperties::TransducerType transducerType, double depth);
		USImageProperties(const USImageProperties& a);

	public:
		/////////////////////////////////////////////////////////////////////
		// simple setters for defining properties
		/////////////////////////////////////////////////////////////////////
		void setImageType(USImageProperties::ImageType imageType);	// Defines the type of information contained in the image
		void setImageState(USImageProperties::ImageState imageState);	// Describes the state the image is currently in
		void setTransducerType(USImageProperties::TransducerType transducerType);	// Defines the type of transducer
		void setScanlineLayout(vec2s scanlineLayout);	// number of scanlines acquired
		void setNumSamples(size_t numSamples);		// number of samples acquired on each scanline
		void setDepth(double depth);					// depth covered
		void setImageResolution(double resolution);  // the resolution of the scanConverted image


		void setScanlineInfo(std::shared_ptr<std::vector<std::vector<ScanlineRxParameters3D> > > scanlines);

		template <typename valueType>
		void setSpecificParameter(std::string parameterName, valueType value);	// set one interface-specific parameter

		/////////////////////////////////////////////////////////////////////
		// simple getters
		/////////////////////////////////////////////////////////////////////
		USImageProperties::ImageType getImageType() const;				// Defines the type of information contained in the image
		USImageProperties::ImageState getImageState() const;			// Describes the state the image is currently in
		USImageProperties::TransducerType getTransducerType() const;	// Defines the type of transducer
		size_t getNumScanlines() const;			// number of scanlines acquired
		vec2s getScanlineLayout() const;
		size_t getNumSamples() const;				// number of samples acquired on each scanline
		double getDepth() const;					// depth covered


		std::shared_ptr<std::vector<std::vector<ScanlineRxParameters3D> > > getScanlineInfo() const;

		bool  hasSpecificParameter(std::string parameterName) const;					// whether one interface-specific parameter exists
		const std::string&  getSpecificParameter(std::string parameterName) const;	// get one interface-specific parameter
		const std::map<std::string, std::string>&  getSpecificParameters() const;	// map to the interface-specific parameters

		/////////////////////////////////////////////////////////////////////
		// Dependent properties, i.e. they only have a getter that computes the return value
		/////////////////////////////////////////////////////////////////////
		double getSampleDistance() const;		// distance between samples of the scanlines
		double getImageResolution() const;		// spatial resolution of image
		bool is2D() const;

	private:
		/////////////////////////////////////////////////////////////////////
		// Defining properties
		/////////////////////////////////////////////////////////////////////
		USImageProperties::ImageType m_imageType;			// Defines the type of information contained in the image
		USImageProperties::ImageState m_imageState;			// Describes the state the image is currently in
		USImageProperties::TransducerType m_transducerType;	// Defines the type of transducer
		size_t m_numScanlines;				// number of scanlines acquired
		vec2s m_scanlineLayout;
		size_t m_numSamples;					// number of samples acquired on each scanline
		double m_depth;						// depth covered
		bool m_imageResolutionSet;			// whether explicit image resolution has been set
		double m_imageResolution;			// explicit image resolution


		// Map for interface specific parameters, they do not define the image itself but its meaning
		std::map<std::string, std::string> m_specificParameters;

		std::shared_ptr<std::vector<std::vector<ScanlineRxParameters3D> > > m_scanlines;
	};

	template<typename valueType>
	inline void USImageProperties::setSpecificParameter(std::string parameterName, valueType value)
	{
		m_specificParameters[parameterName] = stringify(value);
	}

	template<>
	inline void USImageProperties::setSpecificParameter<std::string>(std::string parameterName, std::string value)
	{
		m_specificParameters[parameterName] = value;
	}
}

#endif //!__USIMAGEPROPERTIES_H__
