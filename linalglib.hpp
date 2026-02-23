// Header file for linalglib, a linear algebra library.

# ifndef LINALGLIB_HPP
# define LINALGLIB_HPP

#include <vector>
#include <stdexcept>
#include <cmath>
#include <utility>
#include <complex>

namespace linalglib {

    /**
     * @class Matrix
     * @brief A simple matrix class that stores data in a 1D vector. Stored in row-major order.
     * Designed to be easily understood, rather than hyper-optimized.
     */
    template <typename T>
    class Matrix {
        private:
            std::vector<T> data;
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
            Matrix(std::vector<T> data, size_t rows, size_t cols);

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
            T& operator()(size_t i, size_t j);

            /**
             * @brief Accesses the element at the given row and column indices (for const).
             * @param i Row index (0-based).
             * @param j Column index (0-based).
             * @return The value of the element at the specified position.
             */
            const T& operator()(size_t i, size_t j) const;

            static Matrix<T> identity(size_t n);
    };

    /**
     * @brief Computes the inner product of two vectors.
     * @param a First vector.
     * @param b Second vector.
     * @return The inner product of the two vectors.
     */
    template <typename T>
    T innerProduct(const std::vector<T>& a, std::vector<T> b);

    /**
     * @brief Computes the L2 norm of a vector.
     * @param v The vector for which to compute the norm.
     * @return the L2 norm of the input vector.
     */
    template <typename T>
    double norm(const std::vector<T>& v);

    /**
     * @brief Multiplies a vector by a matrix on the left.
     * @param a The matrix to multiply with.
     * @param x The vector to be multiplied.
     * @return The result of the matrix-vector multiplication.
     */
    template <typename T>
    std::vector<T> matvec(const Matrix<T>& a, const std::vector<T>& x);

    /**
     * @brief Multiplies two matrices together, in the order AB.
     * @param a The first matrix (A).
     * @param b The second matrix (B).
     * @return The result of the matrix multiplication AB.
     */
    template <typename T>
    Matrix<T> matmul(const Matrix<T>& a, const Matrix<T>& b);

    /**
     * @brief Computes the conjugate transpose of a matrix.
     * @param a The matrix to be transposed.
     * @return The conjugate transpose of the input matrix.
     */
    template <typename T>
    Matrix<T> conjugateTranspose(const Matrix<T>& a);
    
    /**
     * @brief Computes the transpose of a matrix.
     * @param a The matrix to be transposed.
     * @return The transpose of the input matrix.
     */
    template <typename T>
    Matrix<T> transpose(const Matrix<T>& a);

    /**
     * @brief Retrieves the columns of a matrix as a vector of vectors.
     * @param a The matrix from which to retrieve the columns.
     * @return a vector of vectors; the nth inner vector is the nth column.
     */
    template <typename T>
    std::vector<std::vector<T>> getColumns(const Matrix<T>& a);

    /**
     * @brief Computes the QR decomposition of a matrix.
     * @param a The matrix to be decomposed.
     * @return The QR decomposition of the input matrix, returned as a tuple of matrices (Q, R).
     */
    template <typename T>
    std::pair<Matrix<T>, Matrix<T>> qrDecomposition(const Matrix<T>& a);

    /**
     * @brief Computes the eigenvalues and eigenvectors of a matrix using the QR algorithm.
     * @param a The matrix for which to compute the eigenvalues and eigenvectors.
     * @param iterations The number of iterations to perform in the QR algorithm (default is 100).
     * @return The pair (vector, matrix) where the vector contains eigenvalues and the matrix is the corresponding eigenvectors (each column).
     */
    template <typename T>
    std::pair<std::vector<T>, Matrix<T>> findEigen(Matrix<T> a, int iterations = 100);

    /**
     * @brief Computes the singular value decomposition (SVD) of a matrix.
     * @param a The matrix to be decomposed.
     * @return the SVD of the input matrix, returned as a tuple of three matrices (U, S, V^T).
     */
    template <typename T>
    std::tuple<Matrix<T>, Matrix<double>, Matrix<T>> svd(const Matrix<T>& a);

    /**
     * @brief Computes the truncated singular value decomposition (SVD) of a matrix.
     * This keeps only the top k largest singular values and their corresponding singular vectors.
     * @param a The matrix to be decomposed.
     * @param k The number of singular values and vectors to keep.
     * @return the truncated SVD of the input matrix, returned as the tuple (U_k, S_k, V_k^T).
     */
    template <typename T>
    std::tuple<Matrix<T>, Matrix<double>, Matrix<T>> svdTruncated(const Matrix<T>& a, size_t k);
}

#endif