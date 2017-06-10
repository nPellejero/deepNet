#!/bin/bash
cant=$3
cd $1
echo  "mirroring folder $1"
echo "-------------------"
for original in $(ls *.$2 | head -$cant)
do
	sinExt=${original%%.*}
	sinExtDotPng="$sinExt.mirror.$2"
	echo "mirroring $original to $sinExtDotPng " 
	convert $original -flop $sinExtDotPng
done
echo "done!"
echo "-------------------"
cd ..
