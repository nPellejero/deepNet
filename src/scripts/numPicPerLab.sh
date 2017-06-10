#!/bin/bash
echo     "------------------------"
echo  -e "-- folder \t numPics "
echo     "------------------------"
for dir in $(ls -d */)
do
	echo -e "-- $dir \t $(ls $dir | wc -l) " 
done
echo      "------------------------"
