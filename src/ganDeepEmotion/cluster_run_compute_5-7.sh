#!/bin/bash

# TMP="/state/partition1/$USER/$$"
# mkdir -p $TMP
# rm -R -f "$TMP/*"
#
# export THEANO_FLAGS="floatX=float32,device=gpu0,config.base_compiledir=\"$TMP\",nvcc.fastmath=True"

# Cambia al directorio actual
#$ -cwd

#$ -j y

# Exporta las variables de entorno:
#$ -V

##$ -o logs
##$ -e logs

# Pide la gpu, uso exclusivo:
#$ -l gpu=true

# El nombre del job
#$ -N "run"

# Selecciono la cola qmla. Usar qmla@compute-0-5 para pedir ese nodo exclusivamente.
##$ -q qmla
#$ -q qmla@compute-0-5,qmla@compute-0-6,qmla@compute-0-7

echo ipython -- $@
ipython -- $@
