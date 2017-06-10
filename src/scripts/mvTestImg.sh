#!/bin/bash
emoc=$1
pwd
for original in $(ls)
do
	soloExt=${original##*_}
	nuevo="$emoc$soloExt"
	echo "moving $original tu $nuevo"
	mv $original $nuevo 
done
echo "done!"
echo "-------------------"

