#!/bin/bash

for folder in $(ls -d */)
do
	cd $folder
	echo  "copiando imagenes en $folder"
	for picture in $(ls)
	do
		echo $picture
		cp $picture /home/npellejero/disruptive/BDEmociones/BDConjunta/
	done
	cd ..
echo "done!"
echo "-------------------"
done
