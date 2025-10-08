CC = clang

FRAMEWORKS = -framework Foundation -framework Metal -framework MetalPerformanceShaders

CFLAGS = -c -fobjc-arc -fmodules -O2 -Wall

all : libmps.a libmps.so

libmps.a: mps.o
	ar rcs $@ $^

libmps.so: mps.o
	$(CC) -shared -o $@ $^ $(FRAMEWORKS)

mps.o: mps.m
	$(CC) $(CFLAGS) -o $@ $^
