# Find where we're running from, so we can store generated files here.

ifeq ($(origin MAKEFILE_DIR), undefined) 
MAKEFILE_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
endif 

#$(info $(MAKEFILE_DIR))

CXX := g++ 
NVCC := nvcc
PYTHON_BIN_PATH= python3
CC :=
AR :=
CXXFLAGS :=
LDFLAGS :=
STDLIB :=

# Try to figure out the host system
HOST_OS :=
ifeq ($(OS),Windows_NT)
	HOST_OS = windows
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		HOST_OS := linux
	endif
	ifeq ($(UNAME_S),Darwin)
		HOST_OS := ios
	endif
endif

#HOST_ARCH := $(shell if [[ $(shell uname -m) =~ i[345678]86 ]]; then echo x86_32; else echo $(shell uname -m); fi)
HOST_ARCH=x86_64
TARGET := $(HOST_OS)
TARGET_ARCH := $(HOST_ARCH)

GENDIR := $(MAKEFILE_DIR)/gen/
TGTDIR := $(GENDIR)$(TARGET)_$(TARGET_ARCH)/
OBJDIR := $(TGTDIR)obj/
BINDIR := $(TGTDIR)bin/
LIBDIR := $(TGTDIR)lib/


TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
# Fix TF LDFLAGS issue on macOS.
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))' | sed "s/-l:libtensorflow_framework.1.dylib/-ltensorflow_framework.1/")
#TF_INCLUDES := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIBS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
CXXFLAGS += -fPIC -shared -O2 -std=c++11 $(TF_CFLAGS)
INCLUDES := -I./third_party/cppjieba/deps \
			-I./third_party/cppjieba/include
LDFLAGS += $(TF_LFLAGS) 


# src and tgts
LIB_SRCS := $(wildcard tf_jieba/cc/*.cc)
LIB_OBJS := $(addprefix $(OBJDIR), $(patsubst %.cc, %.o, $(patsubst %.c, %.o, $(LIB_SRCS))))

# lib
SHARED_LIB := tf_jieba/x_ops.so

all: $(SHARED_LIB)

$(OBJDIR)%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@ $(LDFLAGS)

$(SHARED_LIB): $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(CXX) -fPIC -shared -o $@ $^ $(STDLIB) $(LDFLAGS)

.PHONY: clean
clean:
	-rm -rf $(GENDIR)
	-rm -f $(SHARED_LIB)
