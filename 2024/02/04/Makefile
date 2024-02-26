all:   benchmark 
benchmark: benchmarks/benchmark.cpp 
	c++ -std=c++17 -O3 -mavx2 -mavx512vl -mavx512f -Wall -o benchmark benchmarks/benchmark.cpp -Ibenchmarks 

clean:
	rm -f benchmark
