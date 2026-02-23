// Header file for linalglib, a linear algebra library.

# ifndef LINALGLIB_HPP
# define LINALGLIB_HPP

#include <vector>
#include <stdexcept>
#include <cmath>
#include <utility>

namespace linalglib {

    /**
     * @class Matrix
     * @brief A simple matrix class that stores data in a 1D vector. Stored in row-major order.
     * Designed to be easily understood, rather than hyper-optimized.
     */
    class Matrix {
        private:
            std::vector<double> data;
            size_t nrows;
            size_t ncols;

        public:
            /**
             * @brief Constructs a matrix with the given number of rows and columns, initialized to zero.
             * @param rows Number of rows in the matrix.
             * @param cols Number of columns in the matrix.
             */
            Matrix(size_t rows, size_t cols);

            /**
             * @brief Constructs a matrix from a given data vector and dimensions.
             * @param data A vector containing the matrix data in row-major order.
             * @param rows Number of rows in the Matrix.
             * @param cols Number of columns in the Matrix.
             * @throws std::invalid_argument if the size of the data vector does non match rows * cols.
             */
            Matrix(std::vector<double> data, size_t rows, size_t cols);

            /**
             * @brief Returns the number of rows in the matrix.
             * @return The number of rows in the matrix.
             */
            size_t getRows() const;

            /**
             * @brief Returns the number of columns in the matrix.
             * @return The number of columns in the matrix.
             */
            size_t getCols() const;

            /**
             * @brief Accesses the element at the given row and column indices.
             * @param i Row index (0-based).
             * @param j Column index (0-based).
             * @return A reference to the element at the specified position.
             */
            double& operator()(size_t i, size_t j);

            /**
             * @brief Accesses the element at the given row and column indices (for const).
             * @param i Row index (0-based).
             * @param j Column index (0-based).
             * @return The value of the element at the specified position.
             */
            double operator()(size_t i, size_t j) const;
    };

    /**
     * @brief Computes the inner product of two vectors.
     * @param a First vector.
     * @param b Second vector.
     * @return The inner product of the two vectors.
     */
    double innerProduct(const std::vector<double>& a, std::vector<double> b);

    /**
     * @brief Computes the L2 norm of a vector.
     * @param v The vector for which to compute the norm.
     * @return the L2 norm of the input vector.
     */
    double norm(const std::vector<double>& v);

    /**
     * @brief Creates an n x n identity matrix.
     * @param n The size of the identity matrix.
     * @return An n x n identity matrix.
     */
    Matrix createIdentity(size_t n);

    /**
     * @brief Multiplies a vector by a matrix on the left.
     * @param a The matrix to multiply with.
     * @param x The vector to be multiplied.
     * @return The result of the matrix-vector multiplication.
     */
    std::vector<double> matvec(const Matrix& a, const std::vector<double>& x);

    /**
     * @brief Multiplies two matrices together, in the order AB.
     * @param a The first matrix (A).
     * @param b The second matrix (B).
     * @return The result of the matrix multiplication AB.
     */
    Matrix matmul(const Matrix& a, const Matrix& b);

    /**
     * @brief Computes the conjugate transpose of a matrix.
     * @param a The matrix to be transposed.
     * @return The conjugate transpose of the input matrix.
     */
    Matrix conjugateTranspose(const Matrix& a);
    
    /**
     * @brief Computes the transpose of a matrix.
     * @param a The matrix to be transposed.
     * @return The transpose of the input matrix.
     */
    Matrix transpose(const Matrix& a);

    /**
     * @brief Retrieves the columns of a matrix as a vector of vectors.
     * @param a The matrix from which to retrieve the columns.
     * @return a vector of vectors; the nth inner vector is the nth column.
     */
    std::vector<std::vector<double>> getColumns(const Matrix& a);

    /**
     * @brief Computes the QR decomposition of a matrix.
     * @param a The matrix to be decomposed.
     * @return The QR decomposition of the input matrix, returned as a tuple of matrices (Q, R).
     */
    std::pair<Matrix, Matrix> qrDecomposition(const Matrix& a);

    /**
     * @brief Computes the eigenvalues and eigenvectors of a matrix using the QR algorithm.
     * @param a The matrix for which to compute the eigenvalues and eigenvectors.
     * @param iterations The number of iterations to perform in the QR algorithm (default is 100).
     * @return The pair (vector, matrix) where the vector contains eigenvalues and the matrix is the corresponding eigenvectors (each column).
     */
    std::pair<std::vector<double>, Matrix> findEigen(Matrix a, int iterations);

    /**
     * @brief Computes the singular value decomposition (SVD) of a matrix.
     * @param a The matrix to be decomposed.
     * @return the SVD of the input matrix, returned as a tuple of three matrices (U, S, V^T).
     */
    std::tuple<Matrix, Matrix, Matrix> svd(const Matrix& a);

    /**
     * @brief Computes the truncated singular value decomposition (SVD) of a matrix.
     * This keeps only the top k largest singular values and their corresponding singular vectors.
     * @param a The matrix to be decomposed.
     * @param k The number of singular values and vectors to keep.
     * @return the truncated SVD of the input matrix, returned as the tuple (U_k, S_k, V_k^T).
     */
    std::tuple<Matrix, Matrix, Matrix> svdTruncated(const Matrix& a, size_t k);
}

#endif