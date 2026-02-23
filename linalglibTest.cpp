#include <iostream>
#include "linalglib.cpp"
#include <vector>

int main() {
    // Declarations
    linalglib::Matrix A(2, 3);
    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(0, 2) = 3;
    A(1, 0) = 4.0;
    A(1, 1) = 5.0;
    A(1, 2) = 6.0;

    std::vector<double> matrixb = {5, 3, -10, 7, 2, 1};
    linalglib::Matrix B(matrixb, 3, 2);

    std::vector<double> matrixe = {17.0, 117.0, 2005.0, 2004.0};
    linalglib::Matrix E(matrixe, 2, 2);

    std::vector<double> v1 = {1.0, 6.0, -3.0};
    std::vector<double> v2 = {4.0, 0.0, 7.0};

    // 1. Test matrix construction and element access
    std::cout << "Matrix A:" << std::endl;
    for (size_t i = 0; i < A.getRows(); ++i) {
        for (size_t j = 0; j < A.getCols(); ++j) {
            std::cout << A(i, j) << " ";
        }
        std::cout << std::endl;
    }

    // 2. Test inner product
    double ip = linalglib::innerProduct(v1, v2);
    std::cout << "Inner product of v1 and v2 (should be -17): " << ip << std::endl;

    // 3. Test matrix-vector multiplication
    std::vector<double> mv_result1 = linalglib::matvec(A, v1);
    std::cout << "Matrix-vector product A * v1: [" << mv_result1[0] << "  " << mv_result1[1] << "]" << std::endl;

    std::vector<double> mv_result2 = linalglib::matvec(A, v2);
    std::cout << "Matrix-vector product A * v2: [" << mv_result2[0] << "  " << mv_result2[1] << "]" << std::endl;

    // 4. Test matrix-matrix multiplication
    linalglib::Matrix C = linalglib::matmul(A, B);
    std::cout << "Matrix-matrix product AB: " << std::endl;
    for (size_t i = 0; i < C.getRows(); i++) {
        for (size_t j = 0; j < C.getCols(); j++) {
            std::cout << C(i, j) << " ";
        }
        std::cout << std::endl;
    }

    linalglib::Matrix D = linalglib::matmul(B, A);
    std::cout << "Matrix-matrix product BA: " << std::endl;
    for (size_t i = 0; i < D.getRows(); i++) {
        for (size_t j = 0; j < D.getCols(); j++) {
            std::cout << D(i, j) << " ";
        }
        std::cout << std::endl;
    }

    linalglib::Matrix F = linalglib::matmul(B, E);
    std::cout << "Matrix-matrix product BE: " << std::endl;
    for (size_t i = 0; i < F.getRows(); i++) {
        for (size_t j = 0; j < F.getCols(); j++) {
            std::cout << F(i, j) << " ";
        }
        std::cout << std::endl;
    }
}