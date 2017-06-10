#!/bin/bash

inputDir=$pathredes"txtFiles/BDtxtCropCleanWTest/xabBD.txt"
outputFile="../resultadoRedes.txt"
pathredes="/home/npellejero/tesis/AMFED/redes/"
modelDef="modelosNuevos/VGG16/train_val.prototxt"
modelPre=$1
/home/npellejero/instalaciones/caffe-master/python/classify.py $inputDir $outputFile --mean '' --model_def "$pathredes$modelDef" --pretrained_model "$modelPre" 
