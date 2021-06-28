#pragma once

#include <iostream>

struct Shape {
	size_t x, y;

	Shape(size_t x = 1, size_t y = 1);
};

std::ostream& operator<<(std::ostream& out, const Shape&);