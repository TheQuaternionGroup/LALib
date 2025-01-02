#ifndef LAL_MATRIX_HPP
#define LAL_MATRIX_HPP

#include <cstddef>
#include <array>
#include <initializer_list>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <type_traits>

#include "Vector.hpp"

// Matrix class
template <typename T, std::size_t Rows, std::size_t Cols>
class Mat {
    static_assert(std::is_arithmetic_v<T>, "Matrix elements must be of an arithmetic type.");
    static_assert(Rows > 0 && Cols > 0, "Matrix dimensions must be greater than zero.");

private:
    std::array<T, Rows * Cols> elements_; // Row-major order

public:
    // Constructors

    /**
     * Default constructor: initializes all elements to zero.
     */
    Mat() {
        elements_.fill(static_cast<T>(0));
    }

    /**
     * Constructs a matrix with all elements initialized to a specific value.
     * @param val The value to initialize all elements with.
     */
    explicit Mat(const T& val) {
        elements_.fill(val);
    }

    /**
     * Constructs a matrix using an initializer list.
     * Values are filled row by row. If fewer elements are provided, remaining elements are set to zero.
     * If more elements are provided, excess elements are ignored.
     * @param init Initializer list of values.
     */
    Mat(std::initializer_list<T> init) {
        std::size_t i = 0;
        for (auto it = init.begin(); it != init.end() && i < elements_.size(); ++it, ++i) {
            elements_[i] = *it;
        }
        for (; i < elements_.size(); ++i) {
            elements_[i] = static_cast<T>(0);
        }
    }

    // Static Methods

    /**
     * Constructs an identity matrix. Only enabled if Rows == Cols.
     * @return Identity matrix.
     */
    template <std::size_t R = Rows, std::size_t C = Cols>
    static std::enable_if_t<R == C, Mat<T, R, C>> Identity() {
        Mat<T, R, C> idMat;
        for (std::size_t i = 0; i < R; ++i) {
            idMat(i, i) = static_cast<T>(1);
        }
        return idMat;
    }

    /**
     * Constructs a zero matrix. All elements are zero.
     * @return Zero matrix.
     */
    static Mat<T, Rows, Cols> Zero() {
        return Mat<T, Rows, Cols>();
    }

    // Accessors

    /**
     * Accesses the element at the specified row and column with bounds checking.
     * @param r Row index (0-based).
     * @param c Column index (0-based).
     * @return Reference to the element.
     */
    T& operator()(std::size_t r, std::size_t c) {
        if (r >= Rows || c >= Cols) {
            throw std::out_of_range("Matrix indices out of range.");
        }
        return elements_[r * Cols + c];
    }

    /**
     * Accesses the element at the specified row and column with bounds checking (const version).
     * @param r Row index (0-based).
     * @param c Column index (0-based).
     * @return Const reference to the element.
     */
    const T& operator()(std::size_t r, std::size_t c) const {
        if (r >= Rows || c >= Cols) {
            throw std::out_of_range("Matrix indices out of range.");
        }
        return elements_[r * Cols + c];
    }

    /**
     * Returns the number of rows in the matrix.
     * @return Number of rows.
     */
    constexpr std::size_t rows() const noexcept { return Rows; }

    /**
     * Returns the number of columns in the matrix.
     * @return Number of columns.
     */
    constexpr std::size_t cols() const noexcept { return Cols; }

    // Operator Overloads

    /**
     * Checks if two matrices are equal (same dimensions and corresponding elements).
     * @param other The matrix to compare with.
     * @return True if equal, false otherwise.
     */
    bool operator==(const Mat<T, Rows, Cols>& other) const {
        return elements_ == other.elements_;
    }

    /**
     * Checks if two matrices are not equal.
     * @param other The matrix to compare with.
     * @return True if not equal, false otherwise.
     */
    bool operator!=(const Mat<T, Rows, Cols>& other) const {
        return !(*this == other);
    }

    /**
     * Adds two matrices element-wise.
     * @param other The matrix to add.
     * @return The resultant matrix.
     */
    Mat<T, Rows, Cols> operator+(const Mat<T, Rows, Cols>& other) const {
        Mat<T, Rows, Cols> result;
        for (std::size_t i = 0; i < elements_.size(); ++i) {
            result.elements_[i] = elements_[i] + other.elements_[i];
        }
        return result;
    }

    /**
     * Subtracts two matrices element-wise.
     * @param other The matrix to subtract.
     * @return The resultant matrix.
     */
    Mat<T, Rows, Cols> operator-(const Mat<T, Rows, Cols>& other) const {
        Mat<T, Rows, Cols> result;
        for (std::size_t i = 0; i < elements_.size(); ++i) {
            result.elements_[i] = elements_[i] - other.elements_[i];
        }
        return result;
    }

    /**
     * Multiplies the matrix by a scalar.
     * @param scalar The scalar value.
     * @return The resultant matrix.
     */
    Mat<T, Rows, Cols> operator*(const T& scalar) const {
        Mat<T, Rows, Cols> result(*this);
        for (auto& elem : result.elements_) {
            elem *= scalar;
        }
        return result;
    }

    /**
     * Multiplies two matrices.
     * Only enabled if Cols of the first matrix matches Rows of the second matrix.
     * @param other The matrix to multiply with.
     * @return The resultant matrix.
     */
    template <std::size_t OtherCols>
    Mat<T, Rows, OtherCols> operator*(const Mat<T, Cols, OtherCols>& other) const {
        Mat<T, Rows, OtherCols> result;
        for (std::size_t r = 0; r < Rows; ++r) {
            for (std::size_t c = 0; c < OtherCols; ++c) {
                T sum = static_cast<T>(0);
                for (std::size_t k = 0; k < Cols; ++k) {
                    sum += (*this)(r, k) * other(k, c);
                }
                result(r, c) = sum;
            }
        }
        return result;
    }

    /**
     * Multiplies the matrix by a vector.
     * @param vec The vector to multiply.
     * @return The resultant vector.
     */
    template <std::size_t Dim>
    Vec<T, Rows> operator*(const Vec<T, Dim>& vec) const {
        static_assert(Dim == Cols, "Vector dimension must match the number of columns of the matrix.");
        Vec<T, Rows> result(static_cast<T>(0));
        for (std::size_t r = 0; r < Rows; ++r) {
            for (std::size_t c = 0; c < Cols; ++c) {
                result[r] += (*this)(r, c) * vec[c];
            }
        }
        return result;
    }

    /**
     * Multiplies the matrix by a scalar from the left.
     * @param scalar The scalar value.
     * @param mat The matrix to multiply.
     * @return The resultant matrix.
     */
    friend Mat<T, Rows, Cols> operator*(const T& scalar, const Mat<T, Rows, Cols>& mat) {
        return mat * scalar;
    }

    // Matrix Operations

    /**
     * Transposes the matrix (swaps rows and columns).
     * @return The transposed matrix.
     */
    Mat<T, Cols, Rows> transpose() const {
        Mat<T, Cols, Rows> transposed;
        for (std::size_t r = 0; r < Rows; ++r) {
            for (std::size_t c = 0; c < Cols; ++c) {
                transposed(c, r) = (*this)(r, c);
            }
        }
        return transposed;
    }

    /**
     * Calculates the determinant of the matrix using cofactor expansion.
     * Only enabled if Rows == Cols.
     * @return The determinant value.
     */
    template <std::size_t R = Rows, std::size_t C = Cols>
    std::enable_if_t<R == C, T> determinant() const {
        return _determinant(*this);
    }

    /**
     * Checks if the matrix is invertible (i.e., it is square and has a non-zero determinant).
     * Only enabled if Rows == Cols.
     * @return True if invertible, false otherwise.
     */
    template <std::size_t R = Rows, std::size_t C = Cols>
    std::enable_if_t<R == C, bool> invertible() const {
        T det = determinant();
        return det != static_cast<T>(0);
    }

    /**
     * Calculates the inverse of the matrix using Gaussian elimination (RREF).
     * Only enabled if Rows == Cols and the matrix is invertible.
     * @return The inverse matrix.
     */
    template <std::size_t R = Rows, std::size_t C = Cols>
    std::enable_if_t<R == C, Mat<T, Rows, Cols>> inverse() const {
        static_assert(R == C, "Inverse is only defined for square matrices.");
        if (!invertible()) {
            throw std::runtime_error("Matrix is not invertible.");
        }

        // Create augmented matrix [A | I]
        Mat<T, Rows, 2 * Cols> augmented = this->augment(Mat<T, Rows, Cols>::Identity());

        // Perform Reduced Row Echelon Form (RREF)
        augmented.rref();

        // Extract inverse from augmented matrix
        Mat<T, Rows, Cols> inverse;
        for (std::size_t r = 0; r < Rows; ++r) {
            for (std::size_t c = 0; c < Cols; ++c) {
                inverse(r, c) = augmented(r, c + Cols);
            }
        }

        return inverse;
    }

    /**
     * Transforms the matrix into Reduced Row Echelon Form (RREF) using Gaussian elimination.
     */
    void rref() {
        std::size_t lead = 0;
        for (std::size_t r = 0; r < Rows; ++r) {
            if (lead >= Cols)
                return;
            std::size_t i = r;
            while (std::abs((*this)(i, lead)) < std::numeric_limits<T>::epsilon()) {
                i++;
                if (i == Rows) {
                    i = r;
                    lead++;
                    if (lead == Cols)
                        return;
                }
            }

            // Swap rows i and r
            if (i != r)
                swapRows(i, r);

            // Divide row r by the lead element
            T lv = (*this)(r, lead);
            for (std::size_t j = 0; j < Cols; ++j) {
                (*this)(r, j) /= lv;
            }

            // Subtract multiples of the pivot row from all other rows
            for (std::size_t i2 = 0; i2 < Rows; ++i2) {
                if (i2 != r) {
                    T lv2 = (*this)(i2, lead);
                    for (std::size_t j = 0; j < Cols; ++j) {
                        (*this)(i2, j) -= lv2 * (*this)(r, j);
                    }
                }
            }
            lead++;
        }
    }

    /**
     * Prints the matrix to the standard output.
     */
    void print() const {
        for (std::size_t r = 0; r < Rows; ++r) {
            std::cout << "| ";
            for (std::size_t c = 0; c < Cols; ++c) {
                std::cout << std::fixed << std::setprecision(2) << (*this)(r, c) << " ";
            }
            std::cout << "|\n";
        }
    }

    // Helper Methods

    /**
     * Augments the current matrix with another matrix horizontally.
     * Only enabled if the number of rows matches.
     * @param other The matrix to augment.
     * @return The augmented matrix.
     */
    template <std::size_t OtherCols>
    Mat<T, Rows, Cols + OtherCols> augment(const Mat<T, Rows, OtherCols>& other) const {
        Mat<T, Rows, Cols + OtherCols> augmented;
        for (std::size_t r = 0; r < Rows; ++r) {
            for (std::size_t c = 0; c < Cols; ++c) {
                augmented(r, c) = (*this)(r, c);
            }
            for (std::size_t c = 0; c < OtherCols; ++c) {
                augmented(r, c + Cols) = other(r, c);
            }
        }
        return augmented;
    }

private:
    /**
     * Swaps two rows in the matrix.
     * @param r1 Index of the first row (0-based).
     * @param r2 Index of the second row (0-based).
     */
    void swapRows(std::size_t r1, std::size_t r2) {
        for (std::size_t c = 0; c < Cols; ++c) {
            std::swap((*this)(r1, c), (*this)(r2, c));
        }
    }

    /**
     * Recursively calculates the determinant using cofactor expansion.
     * Only enabled if Rows == Cols.
     * @param matrix The matrix.
     * @return The determinant.
     */
    template <std::size_t R = Rows, std::size_t C = Cols>
    static std::enable_if_t<R == C, T> _determinant(const Mat<T, R, C>& matrix) {
        if constexpr (R == 1) {
            return matrix(0, 0);
        }
        else if constexpr (R == 2) {
            return matrix(0, 0) * matrix(1, 1) - matrix(0, 1) * matrix(1, 0);
        }
        else {
            T det = static_cast<T>(0);
            for (std::size_t c = 0; c < C; ++c) {
                if (matrix(0, c) == static_cast<T>(0))
                    continue;
                // Calculate cofactor
                T cofactor = ((c % 2 == 0) ? 1 : -1) * matrix(0, c) * _determinant(matrix.getCofactor(0, c));
                det += cofactor;
            }
            return det;
        }
    }

    /**
     * Generates a cofactor matrix by removing the specified row and column.
     * @param skipRow The row to skip (0-based).
     * @param skipCol The column to skip (0-based).
     * @return The cofactor matrix.
     */
    Mat<T, Rows - 1, Cols - 1> getCofactor(std::size_t skipRow, std::size_t skipCol) const {
        Mat<T, Rows - 1, Cols - 1> cofactorMat;
        std::size_t r = 0;
        for (std::size_t i = 0; i < Rows; ++i) {
            if (i == skipRow)
                continue;
            std::size_t c = 0;
            for (std::size_t j = 0; j < Cols; ++j) {
                if (j == skipCol)
                    continue;
                cofactorMat(r, c) = (*this)(i, j);
                c++;
            }
            r++;
        }
        return cofactorMat;
    }
};

// Type aliases for common matrix types

// Integer matrices
using Mat2i = Mat<int, 2, 2>;
using Mat3i = Mat<int, 3, 3>;
using Mat4i = Mat<int, 4, 4>;
template <std::size_t Rows, std::size_t Cols>
using MatXi = Mat<int, Rows, Cols>;

// Float matrices
using Mat2f = Mat<float, 2, 2>;
using Mat3f = Mat<float, 3, 3>;
using Mat4f = Mat<float, 4, 4>;
template <std::size_t Rows, std::size_t Cols>
using MatXf = Mat<float, Rows, Cols>;

// Double matrices
using Mat2d = Mat<double, 2, 2>;
using Mat3d = Mat<double, 3, 3>;
using Mat4d = Mat<double, 4, 4>;
template <std::size_t Rows, std::size_t Cols>
using MatXd = Mat<double, Rows, Cols>;

#endif // LAL_MATRIX_HPP

