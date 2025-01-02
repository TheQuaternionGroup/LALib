#pragma once

#include <cstddef>
#include <array>
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <algorithm> 

// Vector class
template <typename T, std::size_t Dim, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
class Vec
{
private:
    std::array<T, Dim> data;

    // Cross product specialized for 3D vectors
    template <std::size_t D = Dim>
    std::enable_if_t<D == 3, Vec<T, 3>> cross_product_3D(const Vec<T, 3>& rhs) const
    {
        Vec<T, 3> result;
        result[0] = data[1] * rhs[2] - data[2] * rhs[1];
        result[1] = data[2] * rhs[0] - data[0] * rhs[2];
        result[2] = data[0] * rhs[1] - data[1] * rhs[0];
        return result;
    }

    // Cross product specialized for 7D vectors
    template <std::size_t D = Dim>
    std::enable_if_t<D == 7, Vec<T, 7>> cross_product_7D(const Vec<T, 7>& rhs) const
    {
        Vec<T, 7> result;
        // Assuming a specific 7D cross product formula
        result[0] = data[1] * rhs[3] - data[3] * rhs[1] +
                    data[2] * rhs[6] - data[6] * rhs[2] +
                    data[4] * rhs[5] - data[5] * rhs[4];

        result[1] = data[2] * rhs[4] - data[4] * rhs[2] +
                    data[3] * rhs[5] - data[5] * rhs[3] +
                    data[0] * rhs[6] - data[6] * rhs[0];

        result[2] = data[3] * rhs[0] - data[0] * rhs[3] +
                    data[4] * rhs[1] - data[1] * rhs[4] +
                    data[5] * rhs[2] - data[2] * rhs[5];

        result[3] = data[4] * rhs[0] - data[0] * rhs[4] +
                    data[5] * rhs[1] - data[1] * rhs[5] +
                    data[2] * rhs[3] - data[3] * rhs[2];

        result[4] = data[5] * rhs[0] - data[0] * rhs[5] +
                    data[1] * rhs[3] - data[3] * rhs[1] +
                    data[4] * rhs[2] - data[2] * rhs[4];

        result[5] = data[0] * rhs[1] - data[1] * rhs[0] +
                    data[2] * rhs[4] - data[4] * rhs[2] +
                    data[3] * rhs[5] - data[5] * rhs[3];

        result[6] = data[0] * rhs[2] - data[2] * rhs[0] +
                    data[1] * rhs[5] - data[5] * rhs[1] +
                    data[3] * rhs[4] - data[4] * rhs[3];

        return result;
    }

public:
    // Default constructor: initializes all elements to zero
    Vec()
    {
        data.fill(0);
    }

    // Constructor: initializes all elements to a specific value
    explicit Vec(const T& val)
    {
        data.fill(val);
    }

    // Constructor from initializer list for easy initialization
    Vec(std::initializer_list<T> init)
    {
        if (init.size() != Dim)
        {
            throw std::invalid_argument("Initializer list size does not match vector dimension.");
        }
        std::copy(init.begin(), init.end(), data.begin());
    }

    // Access operators with bounds checking
    T& operator[](std::size_t index)
    {
        return data.at(index);
    }

    const T& operator[](std::size_t index) const
    {
        return data.at(index);
    }

    // Returns the dimension of the vector
    constexpr std::size_t size() const noexcept
    {
        return Dim;
    }

    // Equality operators
    bool operator==(const Vec<T, Dim>& rhs) const
    {
        return std::equal(data.begin(), data.end(), rhs.data.begin());
    }

    bool operator!=(const Vec<T, Dim>& rhs) const
    {
        return !(*this == rhs);
    }

    // Assignment operator (defaulted)
    Vec<T, Dim>& operator=(const Vec<T, Dim>& rhs) = default;

    // Addition
    Vec<T, Dim> operator+(const Vec<T, Dim>& rhs) const
    {
        Vec<T, Dim> result;
        for (std::size_t i = 0; i < Dim; ++i)
        {
            result[i] = data[i] + rhs[i];
        }
        return result;
    }

    // Subtraction
    Vec<T, Dim> operator-(const Vec<T, Dim>& rhs) const
    {
        Vec<T, Dim> result;
        for (std::size_t i = 0; i < Dim; ++i)
        {
            result[i] = data[i] - rhs[i];
        }
        return result;
    }

    // Scalar multiplication
    Vec<T, Dim> operator*(const T& scalar) const
    {
        if (scalar == 0)
        {
            return Vec<T, Dim>(); // Returns a zero vector
        }
        Vec<T, Dim> result;
        for (std::size_t i = 0; i < Dim; ++i)
        {
            result[i] = data[i] * scalar;
        }
        return result;
    }

    // Scalar division
    Vec<T, Dim> operator/(const T& scalar) const
    {
        if (scalar == 0)
        {
            throw std::runtime_error("Cannot divide by zero.");
        }
        Vec<T, Dim> result;
        for (std::size_t i = 0; i < Dim; ++i)
        {
            result[i] = data[i] / scalar;
        }
        return result;
    }

    // Dot product
    double dot(const Vec<T, Dim>& rhs) const
    {
        double sum = 0.0;
        for (std::size_t i = 0; i < Dim; ++i)
        {
            sum += static_cast<double>(data[i]) * static_cast<double>(rhs[i]);
        }
        return sum;
    }

    // Cross product (only for 3D and 7D vectors)
    Vec<T, Dim> cross(const Vec<T, Dim>& rhs) const
    {
        if constexpr (Dim == 3)
        {
            return cross_product_3D(rhs);
        }
        else if constexpr (Dim == 7)
        {
            return cross_product_7D(rhs);
        }
        else
        {
            throw std::runtime_error("Cross product is only defined for 3D and 7D vectors.");
        }
    }

    // Magnitude of the vector
    double magnitude() const
    {
        double sum = 0.0;
        for (const auto& val : data)
        {
            sum += static_cast<double>(val) * static_cast<double>(val);
        }
        return std::sqrt(sum);
    }

    // Normalize the vector
    Vec<T, Dim> normalize() const
    {
        double mag = magnitude();
        if (mag == 0)
        {
            throw std::runtime_error("Cannot normalize a zero vector.");
        }
        return *this / static_cast<T>(mag);
    }

    // Optional: Overload for scalar multiplication from the left
    friend Vec<T, Dim> operator*(const T& scalar, const Vec<T, Dim>& vec)
    {
        return vec * scalar;
    }
};

// Type aliases for integer vectors
template <std::size_t Dim>
using VecXi = Vec<int, Dim>;

using Vec2i = Vec<int, 2>;
using Vec3i = Vec<int, 3>;
using Vec4i = Vec<int, 4>;

// Type aliases for float vectors
template <std::size_t Dim>
using VecXf = Vec<float, Dim>;

using Vec2f = Vec<float, 2>;
using Vec3f = Vec<float, 3>;
using Vec4f = Vec<float, 4>;

// Type aliases for double vectors
template <std::size_t Dim>
using VecXd = Vec<double, Dim>;

using Vec2d = Vec<double, 2>;
using Vec3d = Vec<double, 3>;
using Vec4d = Vec<double, 4>;
