#pragma once

#include <exception>

namespace ip
{

class IpException : public std::exception
{
public:
    IpException()
        : std::exception()
    {
    }

    explicit IpException(char const* const message)
        : std::exception(message)
    {
    }

    //IpException(char const* const message, int id)
    //    : std::exception(message, id)
    //{
    //}

    //IpException(exception const& _Other)
    //    : std::exception()
    //{
    //}

    //IpException& operator=(IpException const& other)
    //{
    //}

    //virtual ~IpException() throw()
    //{
    //}

    //virtual char const* what() const
    //{
    //    return "Unknown exception";
    //}
};

} // namespace ip

