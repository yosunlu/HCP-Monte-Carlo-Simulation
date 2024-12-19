.PHONY: cuda openmp sequential

push:
	git add .
	git commit -m "$(MSG)"
	git push

pull:
	git reset --hard HEAD
	git pull

cuda:
	chmod +x ./scripts/cuda.sh
	sbatch ./scripts/cuda.sh

openmp:
	chmod +x ./scripts/openmp.sh
	sbatch ./scripts/openmp.sh

sequential:
	chmod +x ./scripts/sequential.sh
	sbatch ./scripts/sequential.sh

clean:
	rm barrier_option_pricing_seq
	rm barrier_option_pricing_openmp
	rm barrier_option_pricing_cuda