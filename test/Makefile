CXX = g++
CFLAGS = -std=c++11
LIBS = -lcppunit

#Serial
INCLUDE_SERIAL = -I ../serial
SRC_SERIAL = test.cpp ActivationTest.cpp LayerTest.cpp FFNN4Test.cpp
SRC_SERIAL += ../serial/Layer.cpp
SRC_SERIAL += ../serial/FFNN4.cpp
TARGET_SERIAL = test

#Parallel
INCLUDE_PARALLEL = -I ../parallel
INCLUDE_PARALLEL += $(INCLUDE_SERIAL)
SRC_PARALLEL = test-parallel.cpp LayerPTest.cpp FFNN4PTest.cpp randomize.cpp
SRC_PARALLEL += ../parallel/LayerP.cpp
SRC_PARALLEL += ../parallel/FFNN4P.cpp
SRC_PARALLEL += ../serial/Layer.cpp
SRC_PARALLEL += ../serial/FFNN4.cpp
TARGET_PARALLEL = test-parallel

#Time
INCLUDE_TIME = $(INCLUDE_PARALLEL)
SRC_TIME = test-time.cpp randomize.cpp
SRC_TIME += ../parallel/LayerP.cpp
SRC_TIME += ../parallel/FFNN4P.cpp
SRC_TIME += ../serial/Layer.cpp
SRC_TIME += ../serial/FFNN4.cpp
TARGET_TIME = test-time

# The C++ compiler
ifneq (,$(shell which icc 2>/dev/null))
CXX = icc
endif # icc
	
ifeq ($(shell uname), Linux)
LIBS += -lrt 
endif

all: serial

serial:
	$(CXX) $(CFLAGS) $(INCLUDE_SERIAL) -o $(TARGET_SERIAL) $(SRC_SERIAL) $(LIBS)
	
parallel:	
	$(CXX) $(CFLAGS) -O2 -DNDEBUG $(INCLUDE_PARALLEL) -o $(TARGET_PARALLEL) $(SRC_PARALLEL) -ltbb $(LIBS)

time:
	$(CXX) $(CFLAGS) -O2 -DNDEBUG $(INCLUDE_TIME) -o $(TARGET_TIME) $(SRC_TIME) -ltbb $(LIBS)
