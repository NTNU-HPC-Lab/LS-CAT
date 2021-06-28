#pragma once

constexpr int GRID_DIMENSIONS = 2;

constexpr int MAX_OBJECT_INTERSECTIONS = (2 << GRID_DIMENSIONS); // (2 ^ GRID_DIMENSIONS)

constexpr float CELLSIZE = 1;

constexpr int XSHIFT = 8;
constexpr int YSHIFT = 0;

constexpr int MAX_ITEMS = 1 << XSHIFT; // 2 ^ XSHIFT

constexpr int OBJECT_COUNT = 4;