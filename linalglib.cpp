// linalglib.cpp : A linear algebra library written for educational purposes. This
// library is not perfectly optimized for performance, but is designed instead to be
// easy to read and understand. It was written as a learning exercise for the author.

#include "linalglib.hpp"
#include <vector>
#include <stdexcept>
#include <cmath>
#include <complex>
#include <utility>

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
    T innerProduct(const std::vector<T>& a, std::vector<T> b) {
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
                result(j, i) =  std::conj(a(i, j));
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
        
        // Initialize Q and R with zeros. 
        // R is always a square matrix (cols x cols).
        Matrix<T> q(rows, cols); 
        Matrix<T> r(cols, cols); 

        for (size_t i = 0; i < cols; ++i) {
            
            // 1. Copy the i-th column from 'a' directly into 'q'
            for (size_t k = 0; k < rows; ++k) {
                q(k, i) = a(k, i);
            }

            // 2. Subtract the projection of previously computed columns
            for (size_t j = 0; j < i; ++j) {
                
                // Calculate projection coefficient
                double dotProduct = 0.0;
                for (size_t k = 0; k < rows; ++k) {
                    dotProduct += q(k, i) * q(k, j);
                }

                // STORE IN R: The projection coefficient goes above the diagonal
                r(j, i) = dotProduct;

                // Subtract the projection
                for (size_t k = 0; k < rows; ++k) {
                    q(k, i) -= dotProduct * q(k, j);
                }
            }

            // 3. Normalize the current column
            double normSq = 0.0;
            for (size_t k = 0; k < rows; ++k) {
                normSq += q(k, i) * q(k, i);
            }
            
            double norm = std::sqrt(normSq);
            // STORE IN R: The normalization factor goes exactly on the diagonal
            r(i, i) = norm;
            
            // Prevent division by zero
            if (norm > 1e-12) { 
                for (size_t k = 0; k < rows; ++k) {
                    q(k, i) /= norm;
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
        Matrix<T> u(a.getRows(), a.getRows());
        Matrix<double> s(a.getRows(), a.getCols());
        Matrix<T> vt(a.getCols(), a.getCols());

        // NOT IMPLEMENTED YET
    }

    template <typename T>
    std::tuple<Matrix<T>, Matrix<double>, Matrix<T>> svdTruncated(const Matrix<T>& a, size_t k) {
        Matrix<T> u(a.getRows(), k);
        Matrix<double> s(k, k);
        Matrix<T> vt(k, a.getCols());

        // NOT IMPLEMENTED YET
    }
}