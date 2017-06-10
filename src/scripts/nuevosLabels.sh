#!/bin/bash
#! para ejecutar en el directorio AMFED
for label in $(ls ./AULabels/new*)
do	
	echo $label
	cat $label | ./scripts/colFiller.sh > $label
done

