#pragma once
/**

Tracer error "Enumeration"

*/

#include "Error.h"
#include <stdexcept>

struct TracerError : public ErrorI
{
    public:
        enum Type
        {
            OK,
            // Logical
            NO_LOGIC_SET,
            // General
            CPU_OUT_OF_MEMORY,
            GPU_OUT_OF_MEMORY,
            // Accelerator Related
            UNABLE_TO_CONSTRUCT_ACCELERATOR,
            UNABLE_TO_CONSTRUCT_BASE_ACCELERATOR,
            // ...


            // End
            END
        };

    private:
        Type        type;

    public:
        // Constructors & Destructor
                    TracerError(Type);
                    ~TracerError() = default;

        operator    Type() const;
        operator    std::string() const override;
};

class TracerException : public std::runtime_error
{
    private:
        TracerError          e;

    protected:
    public:
        TracerException(TracerError::Type t)
            : std::runtime_error("")
            , e(t)
        {}
        TracerException(TracerError::Type t, const char* const err)
            : std::runtime_error(err)
            , e(t)
        {}
        operator TracerError() const { return e; };
};

inline TracerError::TracerError(TracerError::Type t)
    : type(t)
{}

inline TracerError::operator Type() const
{
    return type;
}

inline TracerError::operator std::string() const
{
    static constexpr char const* const ErrorStrings[] =
    {
        "OK",
        "No Tracer Logic is set",
        // General
        "CPU is out of memory",
        "GPU is out of memory",
        // Accelerator Related
        "Unable to construct Accelerator",
        "Unable to construct BaseAccelerator"
    };
    static_assert((sizeof(ErrorStrings) / sizeof(const char*)) == static_cast<size_t>(TracerError::END),
                  "Enum and enum string list size mismatch.");

    return ErrorStrings[static_cast<int>(type)];
}