#include "LALib.hpp"
#include <iostream>

int main() {
    // Create a 3x3 identity matrix
    Mat3f identity = Mat<float, 3, 3>::Identity();
    std::cout << "Identity Matrix:\n";
    identity.print();

    // Create a 3x3 matrix with specific values
    Mat3f mat = {
        3.0f, 0.0f, 2.0f,
        2.0f, 0.0f, -2.0f,
        0.0f, 1.0f, 1.0f
    };
    std::cout << "\nMatrix:\n";
    mat.print();

    // Calculate determinant
    float det = mat.determinant();
    std::cout << "\nDeterminant: " << det << "\n";

    // Calculate inverse
    if (mat.invertible()) {
        Mat3f inv = mat.inverse();
        std::cout << "\nInverse Matrix:\n";
        inv.print();

        // Verify inverse
        // A * A^(-1) = I
        Mat3f result = mat * inv;
        std::cout << "\nResult of Matrix Inversion:\n";
        result.print();
    } else {
        std::cout << "\nMatrix is not invertible.\n";
    }

    // Matrix-vector multiplication
    Vec<float, 3> vec = {1.0f, 2.0f, 3.0f};
    Vec<float, 3> result = mat * vec;
    std::cout << "\nResult of Matrix-Vector Multiplication:\n";
    for (std::size_t i = 0; i < result.size(); ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << "\n";

    return 0;
}
