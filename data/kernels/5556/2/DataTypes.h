#pragma once

#include <windows.h>
#include <gl/GL.h>
#include "glm/glm/vec2.hpp"
#include "Constants.h"

typedef struct CellIdItem
{
	GLuint Cells[MAX_OBJECT_INTERSECTIONS]; //Phantom cell
	GLuint ObjectId;
} CellIdItem;

typedef struct ControlBitsItem
{
	GLubyte HCellType;
	GLubyte PCellTypes;
	GLuint ObjectId;
} ControlBitsItem;

typedef struct GridBox
{
	glm::vec2 min;
	glm::vec2 max;
};

typedef struct Circle
{
	glm::vec2 center;
	float radius;
} Circle;