Run the following command in windows power shell to compile, make sure you have nvcc installed in your machine:
	nvcc -o barrier_option_pricing main.cpp kernel.cu -lcurand

Or simply download the executable file to run the latest build 
	- win11, i7-12700k, rtx4070 - 2024.11.01

Reference:
	https://www.quantstart.com/articles/Monte-Carlo-Simulations-In-CUDA-Barrier-Option-Pricing/