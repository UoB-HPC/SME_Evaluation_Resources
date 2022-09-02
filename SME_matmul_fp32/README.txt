TO COMPILE :

Compiler = armclang 22.0.2

`armclang -static -march=armv8.4-a+sve -Wall -O3 -o sme_matmul_fp32 main.c matmul.c`