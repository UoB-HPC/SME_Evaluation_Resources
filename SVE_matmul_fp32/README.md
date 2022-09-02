# __<u>SVE Matrix Multiply (FP32)</u>__
A simple mini-app which executes a custom number of iterations of a matrix multiply, optimised through the use of Arm Scalable Vector Extension and unrolling the inner-most loop.  
This code has been adapted from the [SVE programming example B5.1](https://developer.arm.com/documentation/dai0548/latest.) from Arm Ltd.  

Performs $C=A*B$ where :
 - $A$ is the left input matrix with dimensions $(M$ x $K)$ 
 - $B$ is the right input matrix with dimenstions $(K$ x $N)$
 - $C$ is the output matrix with dimensions $(M$ x $N)$  


## __<u>Compilation</u>__
Compilation has been tested using Armclang 22.0.2. Static compilation has been used for compatibility with [SimEng](https://github.com/UoB-HPC/SimEng).  
Two main compilation options exist :
 - __Full functionality__ : `armclang -static -march=armv8.4-a+sve -Wall -O3 -o sve_matmul_fp32 main.c matmul.c`
   -  The full mini-app is compiled with all input arguments being utilised.
   - Both the naive verification and SVE matrix multiplys are performed.
 - __Reference functionality__ : `armclang -static -march=armv8.4-a+sve -Wall -O3 -o sve_matmul_fp32_REF main_REF.c matmul.c`
   - Only the naive verification matrix multiplication is performed; i.e. the *iterations* runtime argument will be ignored.
   - Developed to allow for the measurement of non-SVE matrix multiplication cycles / instructions.

## __<u>Execution</u>__
`./sve_matmul_fp32 iterations M K N`
 - __*iterations*__ - number of times the SVE matrix multiplication will be performed *(Default = 1)*.
 - __*M*__ - Number of rows of left input matrix (Default = 128).
 - __*K*__ - Number of columns of left input matrix __*AND*__ number of rows of right input matrix (Default = 128).
 - __*N*__ - Number of columns of right input matrix (Default = 128).
 
 __All dimensions must be a multiple of 16, as well as greater-than or equal to 32.__