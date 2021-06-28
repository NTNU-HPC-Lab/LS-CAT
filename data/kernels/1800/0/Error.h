#pragma once
/**

Generic Error

*/

#include <cstdint>
#include <string>

struct ErrorI
{
    public:
        virtual                 ~ErrorI() = default;
        // Interface
        virtual operator        std::string() const = 0;
};