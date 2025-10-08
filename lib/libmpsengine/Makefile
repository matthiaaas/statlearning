CC = clang

FRAMEWORKS = -framework Foundation -framework Metal -framework MetalPerformanceShaders

CFLAGS = -c -fobjc-arc -fmodules -O2 -Wall

all : libmpsengine.a libmpsengine.so

libmpsengine.a: mps_engine.o
	ar rcs $@ $^

libmpsengine.so: mps_engine.o
	$(CC) -shared -o $@ $^ $(FRAMEWORKS)

mps_engine.o: mps_engine.m
	$(CC) $(CFLAGS) -o $@ $^

main: mps_engine.o
	$(CC) -o $@ $^ $(FRAMEWORKS)

clean:
	rm -f *.o *.a *.so main

.PHONY: all clean
