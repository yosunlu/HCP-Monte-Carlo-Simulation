push:
	git add .
	git commit -m "update script"
	git push

cuda:
	chmod +x cuda.sh
	./cuda.sh