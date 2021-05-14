
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h" //Host executed functions for generating numbers on the GPU
#include "curand_kernel.h" //Device executed functions for generating numbers on the GPU

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <tuple>
#include <string>
#include <functional>
#include <cstdarg>

//template<typename T>
//concept Number = std::is_arithmetic<T>::value;

bool randomizer_generated = false; //boolean indicating whether or not the randomizer has been setup
unsigned long long seed = 10; //default seed value for randomizer
curandGenerator_t generator; //generator for random matrices

//Concepts are not fully supported yet... waiting for C++20 to be more well supported...

//template <Number N>
template <typename N>
struct Matrix {
    unsigned r;
    unsigned c;
    N* arr;

    /*
    * Constructs an empty Matrix
    *
    * @tparam N type of the values of the Matrix
    */
    __host__ Matrix<N>() {}

    /*
    * Constructs a 1x1 Matrix with value val
    *
    * @param val Value of the primitive in the Matrix
    */
    __host__ Matrix<N>(N val)
    {
        r = 1;
        c = 1;
        *arr = val;
    }

    /*
    * Constructs an r by c Matrix and allocates space for the backing pointer arr
    *
    * @tparam N type of the values of the Matrix
    * @param r # of rows
    * @param c # of columns
    */
    __host__ Matrix<N>(unsigned r, unsigned c)
    {
        this->r = r;
        this->c = c;
        this->arr = new N[r * c];
    }

    /*
    * Constructs an r by c Matrix with a given backing pointer that should be in row major form arr[i][j] = arr[i * c + j]
    *
    * @param r # of rows
    * @param c # of columns
    * @param arr given pointer of values to initalize the Matrix with
    */
    __host__ Matrix<N>(unsigned r, unsigned c, N* arr)
    {
        this->r = r;
        this->c = c;
        this->arr = arr;
    }

    /*
    * Contrsucts an r by c Matrix using a given list of numbers to simplify the process.
    * Here's an example of this for creating a 3 by 2 matrix with elements:
    * [ 5 10]
    * [15 20]
    * [ 3 7]
    * 
    * struct Matrix<int> mat = {3, 2, 6, 5, 10, 15, 20, 3, 7}
    * 
    * @param r # of rows
    * @param c # of columns
    * @param num # of elements you want to add following this parameter
    * @param ... List of elements you want to add formatted such that a_{1}, a_{2}, ... , a_{num}
    */
    __host__ Matrix<N>(unsigned r, unsigned c, int num, ...)
    {
        this->r = r;
        this->c = c;
        arr = new N[r * c];

        std::va_list valist;
        va_start(valist, num);
        for (int i = 0; i < num && i < r * c; i++)
        {
            arr[i] = va_arg(valist, N);
        }
        va_end(valist);
    }

    __host__ Matrix(const Matrix& mat) 
    {
        r = mat.r;
        c = mat.c;
        arr = mat.arr;
    }

    /*
    * Automatically allocates memory on the device and based on the parameter copies the backing array from this Matrix onto the device
    * 
    * @param copy If copy is true copies the backing array from this matrix onto the device
    * @return The device matrix
    */
    __host__ inline Matrix<N> cudaSetup(bool copy)
    {
        struct Matrix<N> dev;
        dev.r = r;
        dev.c = c;
        const size_t size = static_cast<unsigned long long>(r) * c * sizeof(N);
        cudaMalloc(&dev.arr, size);
        if (copy)
        {
            cudaMemcpy(dev.arr, this->arr, size, cudaMemcpyHostToDevice);
        }
        return dev;
    }

    /*
    * Access value in the Matrix at point [i][j]
    * 
    * @param i row i
    * @param j column j
    * @return Value at index [i][j]
    */
    __host__ __device__ inline N& operator()(int i, int j) 
    {
        return arr[i * c + j];
    }

    /*
    * Gives a pointer to the head of the ith row in the Matrix
    * 
    * @param i The ith row
    * @return A pointer to the ith row
    */
    __host__ __device__ inline N* operator[](int i) 
    {
        N* ptr = &arr[i * c];
        return ptr;
    }

    __host__ Matrix<N> operator+(const Matrix<N>& o)
    {
        return MatAdd(*this, o);
    }

    __host__ void operator+=(const Matrix<N>& o)
    {
        (*this) = (*this) + o;
    }

    /*
    * Prints the matrix
    */
    __host__ void print()
    {
        for (unsigned i = 0; i < r; i++)
        {
            std::cout << '[';
            for (unsigned j = 0; j < c; j++)
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

/*
* Calculates how many threads per block and the dimension such that: # of threads >= rows * cols
* 
* @param dimBlock Given cuda block to modify. Represents the threads per block
* @param dimGrid Given cuda grid to modify. Represents the amount of blocks
* @param Number of rows you want to base the calculation off
* @param Number of columns you want to base the calculation off
*/
__host__ void threadCalc(dim3& dimBlock, dim3& dimGrid, unsigned rows, unsigned cols)
{
    if (rows > 32 && cols > 32)
    {
        dimBlock.x = 32;
        dimBlock.y = 32;
    }
    else if (rows <= 32 && cols <= 32)
    {
        dimBlock.y = (unsigned int)pow(2, floor(log2(rows)));
        dimBlock.x = (unsigned int)pow(2, floor(log2(cols)));
    }
    else if (rows < 32)
    {
        dimBlock.y = (unsigned int)pow(2, floor(log2(rows)));
        dimBlock.x = cols > 32 * (32 / dimBlock.y) ? 32 * (32 / dimBlock.y) : (unsigned int)pow(2, floor(log2(cols)));
    }
    else // cols < 32
    {
        dimBlock.x = (unsigned int)pow(2, floor(log2(cols)));
        dimBlock.y = rows > 32 * (32 / dimBlock.x) ? 32 * (32 / dimBlock.x) : (unsigned int)pow(2, floor(log2(rows)));
    }

    dimGrid.x = (cols / dimBlock.x) * dimBlock.x == cols ? cols / dimBlock.x :  cols / dimBlock.x + 1;
    dimGrid.y = (rows / dimBlock.y) * dimBlock.y == rows ? rows / dimBlock.y :  rows / dimBlock.y + 1;
}

/*
* Creates a matrix containing doubles from [0,1)
* 
* @param rows Amount of rows the matrix will have
* @param cols Amount of columns the matrix will have
* @return Matrix of type double
*/
__host__ Matrix<double> randomMatrix(unsigned int rows, unsigned int cols)
{
    struct Matrix<double> mat = { rows, cols };
    double* d_A;
    size_t n = static_cast<unsigned long long>(rows) * cols * sizeof(double);

    if (!randomizer_generated)
    {
        std::fstream file;
        file.open("seed.txt", std::ios::in);
        if (file.is_open())
        {
            file >> seed;
            file.close();
        }

        if (seed < 0)
        {
            seed = 0;
        }
        unsigned long long nextSeed;
        if (seed > LLONG_MAX - 5)
        {
            nextSeed = 0;
        }
        else
        {
            nextSeed = seed + 1;
        }

        file.open("seed.txt", std::ios::out);
        if (file.is_open())
        {
            file.clear();
            file << nextSeed;
        }
        curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(generator, seed);
        randomizer_generated = true;
    }
    cudaMalloc(&d_A, n);
    curandGenerateUniformDouble(generator, d_A, static_cast<size_t>(rows) * cols);

    cudaMemcpy(mat.arr, d_A, n, cudaMemcpyDeviceToHost);
    cudaFree(d_A);

    return mat;
}

template <typename N>
__global__ void transposeK(Matrix<N> A, Matrix<N> B)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B.r && j < B.c)
    {
        B[i][j] = A[j][i];
    }
}

/*
* Takes the tranpose of a given matrix
* 
* @param A The given matrix to take the transpose of
*/
template <typename N>
__host__ Matrix<N> transpose(Matrix<N> A)
{
    struct Matrix<N> d_A = A.cudaSetup(true);
    struct Matrix<N> B = { A.c, A.r };
    struct Matrix<N> d_B = B.cudaSetup(false);

    dim3 dimBlock;
    dim3 dimGrid;
    
    threadCalc(dimBlock, dimGrid, B.r, B.c);

    transposeK<N> << <dimGrid, dimBlock >> > (d_A, d_B);

    cudaMemcpy(B.arr, d_B.arr, static_cast<unsigned long long>(B.r) * B.c * sizeof(N), cudaMemcpyDeviceToHost);

    cudaFree(d_B.arr);
    cudaFree(d_A.arr);

    return B;
}

template <typename N>
__global__ void kroneckerProductK(Matrix<N> A, Matrix<N> B, Matrix<N> C)
{
    //todo Speed this up with shared memory
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < C.r && j < C.c)
    {
        C[i][j] = A[i / A.r][j / A.c] * B[i % B.r][j % B.c];
    }
    C(i, j);

}

/*
* Takes the kronecker product (more often called the tensor product) of two matrices. In the order A x B
* Note: The kronecker product of two matrices is not commutative!
* 
* @param A The first matrix
* @param B THe second matrix
*/
template <typename N>
__host__ Matrix<N> kroneckerProduct(Matrix<N> A, Matrix<N> B)
{
    struct Matrix<N> C = { A.r * B.r, A.c * B.c };
    size_t size = static_cast<size_t>(C.r) * C.c * sizeof(N);

    struct Matrix<N> d_A = A.cudaSetup(true);
    struct Matrix<N> d_B = B.cudaSetup(true);
    struct Matrix<N> d_C = C.cudaSetup(false);

    dim3 dimBlock;
    dim3 dimGrid;

    threadCalc(dimBlock, dimGrid, C.r, C.c);

    kroneckerProductK << <dimGrid, dimBlock >> > (d_A, d_B, d_C);

    cudaMemcpy(C.arr, d_C.arr, size, cudaMemcpyDeviceToHost);

    cudaFree(d_C.arr);
    cudaFree(d_B.arr);
    cudaFree(d_C.arr);

    return C;
}

//template <Number N>
template <typename N>
__global__ void MatAddK(Matrix<N> A, Matrix<N> B, Matrix<N> C)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < A.r && j < A.c)
    {
        C[i][j] = A[i][j] + B[i][j];
    }
}

//template <Number N>

/*
* Performs standard matrix addition, where each element is added where C_{i,j} = A_{i,j} + B_{i,j}
* 
* @param A Matrix
* @param B Matrix
*/
template <typename N>
__host__ Matrix<N> MatAdd(Matrix<N> A, Matrix<N> B)
{
    if (A.r != B.r || A.c != B.c)
    {
        throw std::invalid_argument("The dimension of the matrices don't match");
    }
    struct Matrix<N> C = { A.r, A.c };

    size_t size = static_cast<unsigned long long>(A.r) * A.c * sizeof(N); //All matrices are the same size

    struct Matrix<N> d_A = A.cudaSetup(true);

    struct Matrix<N> d_B = B.cudaSetup(true);

    struct Matrix<N> d_C = C.cudaSetup(false);

    dim3 dimBlock;
    dim3 dimGrid;

    threadCalc(dimBlock, dimGrid, C.r, C.c);

    MatAddK<N><< <dimGrid, dimBlock >> > (d_A, d_B, d_C);

    cudaMemcpy(C.arr, d_C.arr, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A.arr);
    cudaFree(d_B.arr);
    cudaFree(d_C.arr);

    return C;
}

int main()
{
    struct Matrix<int> mat = { 10, 10 };

    for (unsigned i = 0; i < mat.r; i++)
    {
        for (unsigned j = 0; j < mat.c; j++)
        {
            mat[i][j] = i * mat.r + j + 1;
        }
    }

    mat.print();

    mat += mat;
    mat.print();
    (randomMatrix(10,10)).print();
    std::cout << '\n';
    (randomMatrix(10, 10)).print();
    std::cout << '\n';
    transpose(mat).print();
    std::cout << "\n\n";

    struct Matrix<int> alpha = { 2, 2 , 4, 0, 5, 6, 7};
    struct Matrix<int> beta = { 2, 2, 4, 1, 2, 3, 4 };

    beta.print();
    alpha.print();

    std::cout << "\n\n";

    kroneckerProduct(beta, alpha).print();

    return 0;
}
