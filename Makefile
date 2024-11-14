push:
	git add .
	git commit -m "$(MSG)"
	git push

pull:
	git reset --hard HEAD
	git pull

cuda:
	chmod +x cuda.sh
	sbatch ./cuda.sh
