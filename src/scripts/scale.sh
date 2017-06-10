#!/bin/bash
#! para ejecutar en el directorio images-croped , images-croped2 etc.

for folder in $(ls -d */) #  ./$1/*.avi ./$1/*.mov  ./$1/*.webm ./$1/*.flv ./$1/*.mpeg ./$1/*.3gp )
do
	cd $folder
	echo "convirtiendo $folder"
	for img in $(ls *.png)
	do
		avconv  -i $img -vf scale=48:48 $img
	done
	cd ..
done
