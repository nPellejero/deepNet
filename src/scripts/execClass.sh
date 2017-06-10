#!/bin/bash

export PYTHONPATH="$PYTHONPATH:/home/digits/nvidia-caffe/python"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda-7.0/lib64"
inputPic=$1
outputFile=$2
pathredes="/home/npellejero/disruptive/redes/"
modelDef=$3
modelPre=$4
/home/digits/nvidia-caffe/python/classify.py $inputPic $outputFile --gpu --mean '' --model_def "$pathredes$modelDef" --pretrained_model "$pathredes$modelPre"
