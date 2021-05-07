
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <tuple>
#include <string>
#include <functional>

//template<typename T>
//concept Number = std::is_arithmetic<T>::value;

//Concepts are not fully supported yet... though hopefully they will be used eventually!

//template <Number N>
template <typename N>
struct Matrix {
    unsigned r;
    unsigned c;
    N* arr;

    __host__ Matrix<N>(){}

    __host__ Matrix<N>(unsigned r, unsigned c)
    {
        this->r = r;
        this->c = c;
        this->arr = new N[r * c];
    }

    __host__ Matrix(const Matrix& mat) {
        r = mat.r;
        c = mat.c;
        arr = mat.arr;
    }

    __host__ __device__ N &operator()(int i, int j)
    {
        /*if (i < 0 || j < 0 || i > r || j > c)
        {
            throw std::out_of_range("index " + std::to_string(i) + "s " + std::to_string(j) + " is out of bounds");
        }*/
        //Commented out since device code doesn't support error handling
        return arr[i * c + j];
    }

    __host__ __device__ N* operator[](int i)
    {
        /*if (r < i)
        {
            throw std::out_of_range("index out of bounds");
        }*/
        //Commented out since device code doesn't support error handling
        N* ptr = &arr[i * c];
        return ptr;
    }

    __host__ void print()
    {
        for (int i = 0; i < r; i++)
        {
            std::cout << '[';
            for (int j = 0; j < c; j++)
            {
                if (j == c - 1)
                {
                    std::cout << arr[i * c + j];
                }
                else
                {
                    std::cout << arr[i * c + j] << ", ";
                }
            }
            std::cout << "]\n";
        }
    }
};

//template <Number N>
template <typename N>
__global__ void MatAddK(Matrix<N> A, Matrix<N> B, Matrix<N> C)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < A.r && j < A.c)
    {
        //C.arr[i * A.c + j] = A.arr[i * A.c + j] + B.arr[i * A.c + j];
        C[i][j] = A[i][j] + B[i][j];
    }
}

//template <Number N>
template <typename N>
__host__ Matrix<N> MatAdd(Matrix<N> A, Matrix<N> B)
{
    if (A.r != B.r || A.c != B.c)
    {
        throw std::invalid_argument("The dimension of the matrices don't match");
    }

    size_t size = static_cast<unsigned long long>(A.r) * A.c * sizeof(N); //All matrices are the same size


    struct Matrix<N> C = {A.r, A.c};
    struct Matrix<N> d_A; d_A.r = A.r; d_A.c = A.c;
    struct Matrix<N> d_B; d_B.r = B.r; d_B.c = B.c;
    struct Matrix<N> d_C; d_C.r = C.r; d_C.c = C.c;

    cudaMalloc(&d_A.arr, size);
    cudaMemcpy(d_A.arr, A.arr, size, cudaMemcpyHostToDevice);

    cudaMalloc(&d_B.arr, size);
    cudaMemcpy(d_B.arr, B.arr, size, cudaMemcpyHostToDevice);

    cudaMalloc(&d_C.arr, size);

    //std::tuple<dim3, dim3> params = foo(A.r, A.c, MatAddK<N>);
    //MatAddK<< <std::get<0>(params), std::get<1>(params) >> > (d_A, d_B, d_C);

    dim3 dimBlock(16, 16);
    dim3 dimGrid(16, 16);

    MatAddK<N><<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    cudaMemcpy(C.arr, d_C.arr, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A.arr);
    cudaFree(d_B.arr);
    cudaFree(d_C.arr);

    std::cout << C[1][2] << '\n';

    return C;
}

//template<typename Function>
//__host__ std::tuple<dim3, dim3> foo(const unsigned rows, const unsigned cols, Function func)
//{
//    int blockSize;   // The launch configurator returned block size 
//    int minGridSize; // The minimum grid size needed to achieve the 
//                     // maximum occupancy for a full device launch 
//    int gridSizeX;    // The actual grid size needed, based on input size 
//    int gridSizeY;
//
//    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, func, 0, 0);
//    gridSizeX = (rows + blockSize - 1) / blockSize;
//    gridSizeY = (cols + blockSize - 1) / blockSize;
//
//    dim3 threadsPerBlock(blockSize, blockSize);
//    dim3 numBlocks(gridSizeX, gridSizeY);
//
//    return std::make_tuple(numBlocks, threadsPerBlock);
//}

int main()
{
    struct Matrix<int> mat = { 10, 10 };

    for (unsigned i = 0; i < mat.r; i++)
    {
        for (unsigned j = 0; j < mat.c; j++)
        {
            mat[i][j] = i * mat.r + j;
        }
    }

    mat.print();

    MatAdd(mat, mat).print();

    return 0;
}
