CC        := gcc
CFLAGS    := -std=gnu99 -O3 -march=native -funroll-loops -fprefetch-loop-arrays -malign-double -LNO:prefetch -g -pipe -fopenmp -Wall -Werror=implicit-function-declaration -Werror=incompatible-pointer-types
LDFLAGS   := -flto -O3 -g -fopenmp
LIBS      += -lopenblas -fopenmp -lrt -lm