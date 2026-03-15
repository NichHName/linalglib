// linalglib.cpp : A linear algebra library written for educational purposes. This
// library is not perfectly optimized for performance, but is designed instead to be
// easy to read and understand. It was written as a learning exercise for the author.

#include "linalglib.hpp"
#include <vector>
#include <stdexcept>
#include <cmath>
#include <complex>
#include <utility>
#include <algorithm>
#include <type_traits>

namespace linalglib {
    template <typename T>
    Matrix<T>::Matrix(size_t rows, size_t cols) : nrows(rows), ncols(cols) {
        data.resize(rows * cols, 0.0);
    }

    template <typename T>
    Matrix<T>::Matrix(std::vector<T> data, size_t rows, size_t cols) : data(data), nrows(rows), ncols(cols) {
        if (data.size() != rows * cols) {
            throw std::invalid_argument("Data size does not match matrix dimensions.");
        }
    }

    template <typename T>
    size_t Matrix<T>::getRows() const {
        return nrows;
    }

    template <typename T>
    size_t Matrix<T>::getCols() const {
        return ncols;
    }

    template <typename T>
    T& Matrix<T>::operator()(size_t i, size_t j) {
        if (i >= nrows || j >= ncols) {
            throw std::out_of_range("Index out of range.");
        }
        return data[i * ncols + j];
    }

    template <typename T>
    const T& Matrix<T>::operator()(size_t i, size_t j) const {
        if (i >= nrows || j >= ncols) {
            throw std::out_of_range("Index out of range.");
        }
        return data[i * ncols + j];
    }

    template <typename T>
    void displayMatrix(const Matrix<T>& a) {
        for (size_t i = 0; i < a.getRows(); i++) {
            for (size_t j = 0; j < a.getCols(); j++) {
                std::cout << a(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }

    template <typename T>
    void displayVector(const std::vector<T>& v) {
        std::cout << "[";
        for (size_t i = 0; i < v.size(); ++i) {
            std::cout << v[i];
            if (i < v.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    template <typename T>
    std::vector<T> vecAdd(const std::vector<T>& a, const std::vector<T>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Vecotrs must be of the same length.");
        }
        std::vector<T> result(a.size(), 0.0);
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = a[i] + b[i];
        }
        return result;
    }

    template <typename T>
    T innerProduct(const std::vector<T>& a, const std::vector<T>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Vectors must be of the same length.");
        }
        T result = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            result += a[i] * b[i];
        }
        return result;
    }

    template <typename T>
    double norm(const std::vector<T>& v) {
        double ip = innerProduct(v, v);
        return std::sqrt(ip);
    }

    template <typename T>
    Matrix<T> Matrix<T>::identity(size_t n) {
        Matrix<T> I(n, n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                I(i, j) = (i == j) ? T(1) : T(0);
            }
        }
        return I;
    }

    template <typename T, typename U>
    Matrix<std::common_type_t<T, U>> matAdd(const Matrix<T>& a, const Matrix<U>& b) {
        if (a.getRows() != b.getRows() || a.getCols() != b.getCols()) {
            throw std::invalid_argument("Matrices need to be the same size.");
        }
        using r_type = std::common_type_t<T, U>;
        Matrix<r_type> result(a.getRows(), b.getRows());
        for (size_t i = 0; i < a.getRows(); ++i) {
            for (size_t j = 0; j < b.getCols(); ++j) {
                result(i, j) = a(i, j) + b(i, j);
            }
        }
        return result;
    }

    template <typename T>
    std::vector<T> matvec(const Matrix<T>& a, const std::vector<T>& x) {
        if (a.getCols() != x.size()) {
            throw std::invalid_argument("Matrix columns must match vector size.");
        }
        std::vector<T> result(a.getRows(), 0.0);
        for (size_t i = 0; i < a.getRows(); ++i) {
            for (size_t j = 0; j < a.getCols(); ++j) {
                result[i] += a(i, j) * x[j];
            }
        }
        return result;
    }

    template <typename T>
    Matrix<T> matmul(const Matrix<T>& a, const Matrix<T>& b) {
        if (a.getCols() != b.getRows()) {
            throw std::invalid_argument("Matrix A columns must match Matrix B rows.");
        }
        Matrix<T> result(a.getRows(), b.getCols());
        for (size_t i = 0; i < a.getRows(); ++i) {
            for (size_t j = 0; j < b.getCols(); ++j) {
                for (size_t k = 0; k < a.getCols(); ++k) {
                    result(i, j) += a(i, k) * b(k, j);
                }
            }
        }
        return result;
    }

    template <typename T>
    Matrix<T> transpose(const Matrix<T>& a) {
        Matrix<T> result(a.getCols(), a.getRows());
        for (size_t i = 0; i < a.getRows(); ++i) {
            for (size_t j = 0; j < a.getCols(); ++j) {
                result(j, i) = a(i, j);
            }
        }
        return result;
    }

    template <typename T>
    Matrix<T> conjugateTranspose(const Matrix<T>& a) {
        Matrix<T> result(a.getCols(), a.getRows());
        for (size_t i = 0; i < a.getRows(); ++i) {
            for (size_t j = 0; j < a.getCols(); ++j) {
                if constexpr (std::is_same_v<T, std::complex<double>> || std::is_same_v<T, std::complex<float>>) {
                    result(j, i) = std::conj(a(i, j));
                } else {
                    result(j, i) = a(i, j);
                }
            }
        }
        return result;
    }

    template <typename T>
    std::vector<std::vector<T>> getColumns(const Matrix<T>& a) {
        std::vector<std::vector<T>> columns(a.getCols(), std::vector<T>(a.getRows()));
        for (size_t i = 0; i < a.getRows(); ++i) {
            for (size_t j = 0; j < a.getCols(); ++j) {
                columns[j][i] = a(i, j);
            }
        }
        return columns;
    }

    template <typename T>
    std::pair<Matrix<T>, Matrix<T>> qrDecomposition(const Matrix<T>& a) {
        size_t rows = a.getRows();
        size_t cols = a.getCols();

        Matrix<T> q(rows, cols);
        Matrix<T> r(cols, cols);

        // Extract columns into a format compatible with your helper functions
        std::vector<std::vector<T>> columns = getColumns(a);
        std::vector<std::vector<T>> q_columns(cols, std::vector<T>(rows));

        for (size_t i = 0; i < cols; ++i) {
            // Start with the original column vector
            std::vector<T> v = columns[i];

            // Orthogonalize against all previous vectors in the basis
            for (size_t j = 0; j < i; ++j) {
                // Use your innerProduct function to find the projection coefficient
                T projection_coeff = innerProduct(q_columns[j], columns[i]);
                
                r(j, i) = projection_coeff;

                // Subtract the projection: v = v - (q_j * proj)
                for (size_t k = 0; k < rows; ++k) {
                    v[k] -= projection_coeff * q_columns[j][k];
                }
            }

            // Use your norm function to find the length of the orthogonalized vector
            double magnitude = norm(v);
            r(i, i) = static_cast<T>(magnitude);

            // Normalize and store back in Q
            if (magnitude > 1e-12) {
                for (size_t k = 0; k < rows; ++k) {
                    T normalized_val = v[k] / static_cast<T>(magnitude);
                    q_columns[i][k] = normalized_val;
                    q(k, i) = normalized_val;
                }
            }
        }

        return {q, r};
    }

    template <typename T>
    std::pair<std::vector<T>, Matrix<T>> findEigen(Matrix<T> a, int iterations) {
        size_t n = a.getRows();
        Matrix<T> v = Matrix<T>::identity(n);

        for (int k = 0; k < iterations; ++k) {
            // 1. Decompose the current matrix A
            auto [q, r] = qrDecomposition(a);

            // 2. Recombine in reverse order: A_{k+1} = R * Q
            a = matmul(r, q);

            // 3. Accumulate the eigenvectors: V_{k+1} = V_k * Q
            v = matmul(v, q);
        }

        // Eigenvalues are on the main diagonal
        std::vector<T> eigenvalues(n);
        for (size_t i = 0; i < n; ++i) {
            eigenvalues[i] = a(i, i);
        }

        // Columns of v are corresponding eigenvectors
        return {eigenvalues, v};
    }

    template <typename T>
    std::tuple<Matrix<T>, Matrix<double>, Matrix<T>> svd(const Matrix<T>& a) {
        size_t m = a.getRows();
        size_t n = a.getCols();

        Matrix<T> u(m, m);
        Matrix<double> s(m, n);
        Matrix<T> vt(n, n);

        Matrix<T> conjTrans = conjugateTranspose(a);

        // Compute A*A
        Matrix<T> a_star_a = matmul(conjTrans, a);

        // Find eigenvalues and eigenvectors of A*A
        // Number of iterations controls the accuracy of the QR algorithm
        int iterations = 100; 
        auto [eigenvalues, v] = findEigen(a_star_a, iterations);

        // V* is the conjugate transpose of V
        vt = conjugateTranspose(v);

        // 4. Extract columns of V to compute U
        std::vector<std::vector<T>> v_cols = getColumns(v);

        // Populate S and calculate U
        for (size_t i = 0; i < n; ++i) {
            // We use std::abs to ensure we don't pass negative numbers to sqrt 
            // due to minor floating-point inaccuracies
            double singular_value = std::sqrt(std::abs(eigenvalues[i]));
            
            // Place singular value on the diagonal of S
            if (i < m) {
                s(i, i) = singular_value;
            }

            // Calculate corresponding column, u_i = (A * v_i) / sigma_i
            if (i < m) {
                if (singular_value > 1e-12) {
                    std::vector<T> u_col = matvec(a, v_cols[i]);
                    for (size_t k = 0; k < m; ++k) {
                        // Cast singular_value to T to match Matrix<T> types
                        u(k, i) = u_col[k] / static_cast<T>(singular_value);
                    }
                } else {
                    // For zero singular values, set a standard basis vector 
                    // (A simplification for educational purposes)
                    u(i, i) = T(1.0); 
                }
            }
        }

        return {u, s, vt};
    }

    template <typename T>
    std::tuple<Matrix<T>, Matrix<double>, Matrix<T>> svdTruncated(const Matrix<T>& a, size_t k) {
        size_t m = a.getRows();
        size_t n = a.getCols();

        // Safety check: k shouldn't exceed the matrix dimensions
        k = std::min({k, m, n});

        Matrix<T> u(m, k);
        Matrix<double> s(k, k);
        Matrix<T> vt(k, n);

        Matrix<T> conjTrans = conjugateTranspose(a);

        // Compute A*A
        Matrix<T> a_star_a = matmul(conjTrans, a);

        // Find eigenvalues and eigenvectors of A*A
        int iterations = 100; 
        auto [eigenvalues, v] = findEigen(a_star_a, iterations);

        // 3. Extract only the first k columns of V to form VT (k x n matrix)
        for (size_t i = 0; i < k; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if constexpr (std::is_same_v<T, std::complex<double>> || std::is_same_v<T, std::complex<float>>) {
                    vt(i, j) = std::conj(v(j, i));
                } else {
                    vt(i, j) = v(j, i);
                }
            }
        }

        // Extract columns of V to compute U
        std::vector<std::vector<T>> v_cols = getColumns(v);

        // 5. Populate S and calculate U for only the first k components
        for (size_t i = 0; i < k; ++i) {
            double singular_value = std::sqrt(std::abs(eigenvalues[i]));
            
            // Place singular value on the diagonal of S
            s(i, i) = singular_value;

            // Calculate corresponding column of U: u_i = (A * v_i) / sigma_i
            if (singular_value > 1e-12) {
                std::vector<T> u_col = matvec(a, v_cols[i]);
                for (size_t row = 0; row < m; ++row) {
                    u(row, i) = u_col[row] / static_cast<T>(singular_value);
                }
            } else {
                // For zero singular values, set a standard basis vector 
                if (i < m) {
                    u(i, i) = T(1.0); 
                }
            }
        }

        return {u, s, vt};
    }
}