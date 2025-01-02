#include "LALib.hpp"
#include <iostream>

int main()
{
    Vec3f v1 = {1.0f, 2.0f, 3.0f};
    Vec3f v2 = {4.0f, 5.0f, 6.0f};

    Vec3f v3 = v1 + v2;
    Vec3f v4 = v1.cross(v2);
    float dotProduct = v1.dot(v2);
    Vec3f v5 = v1.normalize();

    std::cout << "v3: ";
    for (std::size_t i = 0; i < v3.size(); ++i) std::cout << v3[i] << " ";
    std::cout << "\nCross Product v4: ";
    for (std::size_t i = 0; i < v4.size(); ++i) std::cout << v4[i] << " ";
    std::cout << "\nDot Product: " << dotProduct;
    std::cout << "\nNormalized v1: ";
    for (std::size_t i = 0; i < v5.size(); ++i) std::cout << v5[i] << " ";
    std::cout << std::endl;

    return 0;
}
