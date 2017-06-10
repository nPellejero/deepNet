#!/bin/bash
#! para ejecutar en el directorio congealreal 
for nombres in $(ls x*)
do
	./funnelReal $nombres model.txt $nombres
	echo "listo con $nombres . continuo?"
	read -rn1 a
	if [ "$a" == 'n' ] || [ "$a" == 'N' ]; then
		break
	fi
		 
done
