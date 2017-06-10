#!/bin/bash
cant=$3
cd $1
echo  "mirroring folder $1"
echo "-------------------"
for original in $(ls *.$2 | tail -$cant)
do
	sinExt=${original%%.*}
	sinExtDotPng="$sinExt.color.$2"
	echo "mirroring $original to $sinExtDotPng " 
	convert $original -colorspace HSB $sinExtDotPng
done
echo "done!"
echo "-------------------"
cd ..
