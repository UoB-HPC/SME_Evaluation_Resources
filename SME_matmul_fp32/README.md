# __<u>SME Matrix Multiply (FP32)</u>__
A simple mini-app which executes a custom number of iterations of a matrix multiply, optimised through the use of Arm Scalable Matrix Extension.  
This code has been adapted from an SME programming example provided by Arm Ltd.  

Performs $C=A*B$ via a summation of outer-products, where :
 - $A$ is the left input matrix with dimensions $(M$ x $K)$ 
 - $B$ is the right input matrix with dimenstions $(K$ x $N)$
 - $C$ is the output matrix with dimensions $(M$ x $N)$  

 A pre-process step is taken to transpose matrix $A$, before calling the SME matrix multiply function. This allows for row-major memory access of both input matricies.

 The SME matrix multiply function is written as commented inline assembly, including some hex representations of branches and SME instructions. This was done to allow for compilation to 
 a Linux ELF binary file using current compilers which do not support Arm's Scalable Matrix Extension at the time of creation.


## __<u>Compilation</u>__
Compilation has been tested using Armclang 22.0.2. Static compilation has been used for compatibility with [SimEng](https://github.com/UoB-HPC/SimEng).  
Two main compilation options exist :
 - __Full functionality__ : `armclang -static -march=armv8.4-a+sve -Wall -O3 -o sme_matmul_fp32 main.c matmul.c`
   -  The full mini-app is compiled with all input arguments being utilised.
   - Both the naive verification and SME matrix multiplys are performed.
 - __Reference functionality__ : `armclang -static -march=armv8.4-a+sve -Wall -O3 -o sme_matmul_fp32_REF main_REF.c matmul.c`
   - Only the naive verification matrix multiplication is performed; i.e. the *iterations* runtime argument will be ignored.
   - Developed to allow for the measurement of non-SME matrix multiplication cycles / instructions.

## __<u>Execution</u>__
`./sme_matmul_fp32 iterations M K N`
 - __*iterations*__ - number of times the SME matrix multiplication will be performed *(Default = 1)*.
 - __*M*__ - Number of rows of left input matrix (Default = 128).
 - __*K*__ - Number of columns of left input matrix __*AND*__ number of rows of right input matrix (Default = 128).
 - __*N*__ - Number of columns of right input matrix (Default = 128).
 
 __The K dimension must be greater-than 2.__