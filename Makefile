TARGET := eigvecs

# Select compiler and linker: intel, gnu, clang
COMPILER := gnu

# Set global defines
DEFINES := -DNDEBUG -DALIGNMENT=64

# Define dynamically linked libraries
LIBS := -lrt -lm

# ------------------------------------------------------------------------------
# Selection of flags
# ------------------------------------------------------------------------------

ifeq ($(COMPILER), intel)
	CC        := icc
	CFLAGS    := -Wall -std=gnu99 -O2 -mtune=skylake-avx512 -malign-double -qopt-prefetch -qopenmp -ipo
	LDFLAGS   := -O2 -ipo -qopenmp
	LIBS      += -mkl
else ifeq ($(COMPILER), clang)
	CC        := clang
	CFLAGS    := -Wall -Werror=implicit-function-declaration -O3 -std=gnu99 -pipe -fopenmp
	LDFLAGS   := -flto -O3 -fopenmp -g
	LIBS      += -lopenblas -fopenmp
else # gnu
	CC        := gcc
	CFLAGS    := -std=gnu99 -O3 -march=native -funroll-loops -fprefetch-loop-arrays -malign-double -LNO:prefetch -g -pipe -fopenmp -Wall -Werror=implicit-function-declaration -Werror=incompatible-pointer-types
	LDFLAGS   := -flto -O3 -g -fopenmp
	LIBS      += -lopenblas -fopenmp
endif


# Select all C source files
SRCS := $(wildcard *.c)
OBJS := $(SRCS:.c=.o)


# ------------------------------------------------------------------------------
# Makefile rules and targets
# ------------------------------------------------------------------------------

.SUFFIXES: .c .o

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o $(TARGET) $(LIBS)

%.o : %.c
	$(CC) $(CFLAGS) $(DEFINES) -c $< -o $@

clean: 
	rm -f $(TARGET) *.o


.PHONY: clean
