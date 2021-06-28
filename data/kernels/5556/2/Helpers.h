#pragma once

#include <windows.h>
#include <gl/GL.h>
#include "glm/glm/vec3.hpp"

#include "Constants.h"


glm::uvec2 posToCoords(glm::vec2 pos)
{
	glm::uvec2 coord;
	coord.x = (GLuint)floor(pos.x / CELLSIZE);
	coord.y = (GLuint)floor(pos.y / CELLSIZE);
	return coord;
}

GLubyte posToCellType(glm::vec2 pos)
{
	glm::uvec2 coords = posToCoords(pos);
	return coords.x % 2 * 2 + coords.y % 2;
}

GridBox coordsToGridBox(GLuint xCoord, GLuint yCoord)
{
	GridBox box;
	box.min.x = xCoord * CELLSIZE;
	box.max.x = box.min.x + CELLSIZE;
	box.min.y = yCoord * CELLSIZE;
	box.max.y = box.min.y + CELLSIZE;

	return box;
}

bool collides(Circle circle, GridBox box)
{
	bool xIntersect = (circle.center.x + circle.radius > box.min.x) && (circle.center.x - circle.radius < box.max.x);
	bool yIntersect = (circle.center.y + circle.radius > box.min.y) && (circle.center.y - circle.radius < box.max.y);

	return xIntersect && yIntersect;
}

GLuint posToHash(glm::vec2 pos)
{
	glm::uvec2 coords = posToCoords(pos);
	return (coords.x << XSHIFT) |
		(coords.y << YSHIFT);
}