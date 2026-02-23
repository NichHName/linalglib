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

    Matrix::Matrix(size_t rows, size_t cols) : nrows(rows), ncols(cols) {
        data.resize(rows * cols, 0.0);
    }

    Matrix::Matrix(std::vector<double> data, size_t rows, size_t cols) : data(data), nrows(rows), ncols(cols) {
        if (data.size() != rows * cols) {
            throw std::invalid_argument("Data size does not match matrix dimensions.");
        }
    }

    size_t Matrix::getRows() const {
        return nrows;
    }

    size_t Matrix::getCols() const {
        return ncols;
    }

    double& Matrix::operator()(size_t i, size_t j) {
        if (i >= nrows || j >= ncols) {
            throw std::out_of_range("Index out of range.");
        }
        return data[i * ncols + j];
    }

    double Matrix::operator()(size_t i, size_t j) const {
        if (i >= nrows || j >= ncols) {
            throw std::out_of_range("Index out of range.");
        }
        return data[i * ncols + j];
    }

    double innerProduct(const std::vector<double>& a, std::vector<double> b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Vectors must be of the same length.");
        }
        double result = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            result += a[i] * b[i];
        }
        return result;
    }

    double norm(const std::vector<double>& v) {
        double sumSquares = 0.0;
        for (double val : v) {
            sumSquares += val * val;
        }
        return std::sqrt(sumSquares);
    }

    Matrix createIdentity(size_t n) {
        Matrix I(n, n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                I(i, j) = (i == j) ? 1.0 : 0.0;
            }
        }
        return I;
    }

    std::vector<double> matvec(const Matrix& a, const std::vector<double>& x) {
        if (a.getCols() != x.size()) {
            throw std::invalid_argument("Matrix columns must match vector size.");
        }
        std::vector<double> result(a.getRows(), 0.0);
        for (size_t i = 0; i < a.getRows(); ++i) {
            for (size_t j = 0; j < a.getCols(); ++j) {
                result[i] += a(i, j) * x[j];
            }
        }
        return result;
    }

    Matrix matmul(const Matrix& a, const Matrix& b) {
        if (a.getCols() != b.getRows()) {
            throw std::invalid_argument("Matrix A columns must match Matrix B rows.");
        }
        Matrix result(a.getRows(), b.getCols());
        for (size_t i = 0; i < a.getRows(); ++i) {
            for (size_t j = 0; j < b.getCols(); ++j) {
                for (size_t k = 0; k < a.getCols(); ++k) {
                    result(i, j) += a(i, k) * b(k, j);
                }
            }
        }
        return result;
    }

    Matrix transpose(const Matrix& a) {
        Matrix result(a.getCols(), a.getRows());
        for (size_t i = 0; i < a.getRows(); ++i) {
            for (size_t j = 0; j < a.getCols(); ++j) {
                result(j, i) = a(i, j);
            }
        }
        return result;
    }

    Matrix conjugateTranspose(const Matrix& a) {
        Matrix result(a.getCols(), a.getRows());
        for (size_t i = 0; i < a.getRows(); ++i) {
            for (size_t j = 0; j < a.getCols(); ++j) {
                result(j, i) = a(i, j); // Need to fix this to handle complex numbers
            }
        }
        return result;
    }

    std::vector<std::vector<double>> getColumns(const Matrix& a) {
        std::vector<std::vector<double>> columns(a.getCols(), std::vector<double>(a.getRows()));
        for (size_t i = 0; i < a.getRows(); ++i) {
            for (size_t j = 0; j < a.getCols(); ++j) {
                columns[j][i] = a(i, j);
            }
        }
        return columns;
    }

    std::pair<Matrix, Matrix> qrDecomposition(const Matrix& a) {
        size_t rows = a.getRows();
        size_t cols = a.getCols();
        
        // Initialize Q and R with zeros. 
        // R is always a square matrix (cols x cols).
        Matrix q(rows, cols); 
        Matrix r(cols, cols); 

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

    std::pair<std::vector<double>, Matrix> findEigen(Matrix a, int iterations = 100) {
        size_t n = a.getRows();
        Matrix v = createIdentity(n);

        for (int k = 0; k < iterations; ++k) {
            // 1. Decompose the current matrix A
            auto [q, r] = qrDecomposition(a);

            // 2. Recombine in reverse order: A_{k+1} = R * Q
            a = matmul(r, q);

            // 3. Accumulate the eigenvectors: V_{k+1} = V_k * Q
            v = matmul(v, q);
        }

        // Eigenvalues are on the main diagonal
        std::vector<double> eigenvalues(n);
        for (size_t i = 0; i < n; ++i) {
            eigenvalues[i] = a(i, i);
        }

        // Columns of v are corresponding eigenvectors
        return {eigenvalues, v};
    }

    std::tuple<Matrix, Matrix, Matrix> svd(const Matrix& a) {
        Matrix u(a.getRows(), a.getRows());
        Matrix s(a.getRows(), a.getCols());
        Matrix vt(a.getCols(), a.getCols());

        // NOT IMPLEMENTED YET
    }

    std::tuple<Matrix, Matrix, Matrix> svdTruncated(const Matrix& a, size_t k) {
        Matrix u(a.getRows(), k);
        Matrix s(k, k);
        Matrix vt(k, a.getCols());

        // NOT IMPLEMENTED YET
    }
}