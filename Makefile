.PHONY: openmp cuda

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
