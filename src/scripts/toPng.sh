#!/bin/bash

for dir in $(ls -d */) 
do
	cd $dir
	for original in $(ls)
	do
		sinExt=${original%%.*}
		convert $original $sinExt.png
	done
	cd ..
done
