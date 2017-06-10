#!/bin/bash 
IFS=","
fps=14
read time col
echo $col
oldTime=0
while read time other 
do
	deltaTime=$(echo "$time-$oldTime" | bc)
	numRow=$(echo "$deltaTime*$fps" | bc)
	num=$( printf "%.0f" $numRow)
	for((c=1; c<$num+1; c++)) 
	do
		echo $other
	done
	oldTime=$time
done
