push:
	git add .
	git commit -m "update script"
	git push

reset:
	git reset --hard HEAD
	git pull

cuda:
	chmod +x kernel.cu
	chmod +x cuda.sh
	sbatch ./cuda.sh

