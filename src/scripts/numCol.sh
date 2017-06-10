#!/bin/bash
#! para ejecutar en el directorio AULabels

lineAnt=""
for label in $(ls *label.csv)
do

line=$(head -n 1 $label)
if [ "$line" == "$lineAnt" ]; then
	continue;
fi
lineAnt=$line
echo -e "$line\n"

done
	#echo -e "-- Backward:\t$BackwardT"
