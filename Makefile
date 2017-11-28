SRC_DIR = ./src
INC_DIR = ./include
BOOST_INC_DIR = ../boost/include
BOOST_LIB_DIR = ../boost/lib
OBJ_DIR = ./obj
BIN_DIR = ./bin
DEDISP_DIR = ./dedisp_paf
CC=g++
NVCC=/usr/local/cuda-8.0/bin/nvcc
DEBUG=#-g -G

INCLUDE = -I${INC_DIR} -I${BOOST_INC_DIR}
LIBS = -L${DEDISP_DIR}/lib -L${BOOST_LIB_DIR} -lstdc++ -lboost_system -lpthread -lcudart -lcuda

CFLAGS = -Wall -Wextra -std=c++11
NVCC_FLAG = -gencode=arch=compute_52,code=sm_52 --std=c++11 -lcufft -Xcompiler ${DEBUG} #--default-stream per-thread

CPPOBJECTS = ${OBJ_DIR}/DedispPlan.o

CUDAOBJECTS = ${OBJ_DIR}/lofte.o ${OBJ_DIR}/gpu_pool.o ${OBJ_DIR}/kernels.o ${OBJ_DIR}/dedisp.o ${OBJ_DIR}/main_pool.o


all: lofte
lofte: ${CUDAOBJECTS} ${CPPOBJECTS}
	${NVCC} ${NVCC_FLAG} ${INCLUDE} ${LIBS} ${CUDAOBJECTS} ${CPPOBJECTS} -o ${BIN_DIR}/lofte

${OBJ_DIR}/%.o: ${SRC_DIR}/%.cu
	${NVCC} -c ${NVCC_FLAG} ${INCLUDE} $< -o $@

${OBJ_DIR}/%.o: ${SRC_DIR}/%.cpp
	${CC} -c ${CFLAGS} ${INCLUDE} $< -o $@

.PHONY: clean

clean:
	rm -f ${OBJ_DIR}/*.o ${BIN_DIR}/*
