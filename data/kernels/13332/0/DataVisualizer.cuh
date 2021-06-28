/*
 * DataVisualizer.h
 *
 *  Created on: 21.06.2016
 *      Author: Sebastian Reinhart
 */

#ifndef DATAVISUALIZER_H_
#define DATAVISUALIZER_H_

#include "data.cuh"
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <sstream>
#include <cuda.h>
#include <stdio.h>
#include <map>
#include "PointCellDevice.cuh"

#define CANVASSIZE 2000
#define CANVASFACTOR 10

class DataVisualizer {
public:
	DataVisualizer();
	virtual ~DataVisualizer();

	void visualizeSegmentsAsPointCloud(cartesian_segment* segments, std::string number, int segment_count);
	void visualizeVehiclesAsRectangle(std::vector<std::vector<laserdata_cartesian> > segments, std::string number);
	void visualizeConvoys(std::vector<EMLPos> EML, Convoy* convoys, int startIndexConvoys, int endIndexConvoys);
	void visualizeHistory(std::vector<EMLPos> EML, History* history, int startIndex, int endIndex);

private:
	std::string colors[20];
};

#endif /* DATAVISUALIZER_H_ */
