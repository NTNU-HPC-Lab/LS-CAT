//
// Created by jlebas01 on 04/04/2020.
//

#ifndef HISTOGRAM_PROJECT_TOOLS_HPP
#define HISTOGRAM_PROJECT_TOOLS_HPP
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <vector_types.h>


namespace utils {

    std::ostream &operator<<(std::ostream &os, const uchar4 &c);

    void compareImages(const std::vector<uchar4> &a, const std::vector<uchar4> &b);

}

#endif //HISTOGRAM_PROJECT_TOOLS_HPP
