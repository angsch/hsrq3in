TOPSRCDIR := .
include $(TOPSRCDIR)/make.inc.gnu  # make.inc.intel, make.inc.llvm

# Set global defines
DEFINES := -DNDEBUG -DALIGNMENT=64

# Select all C source files
SRCS := $(wildcard *.c)
OBJS := $(SRCS:.c=.o)


eigvecs: $(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o eigvecs $(LIBS)

%.o : %.c
	$(CC) $(CFLAGS) $(DEFINES) -c $< -o $@

clean: 
	rm -f eigvecs *.o

.PHONY: eigvecs clean
