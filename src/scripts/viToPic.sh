#!/bin/bash
#! para ejecutar en el directorio AMFED
for video in $(ls ./Video-AVI)
do
	nombre=${video%%.*}
	echo "convirtiendo '$nombre' a imagenes"
	mkdir ./images/$nombre
	ffmpeg  -i ./Video-AVI/$nombre.avi -r 14  -vcodec png ./images-png/$nombre/Pictures%05d.png
done
