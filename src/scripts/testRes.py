#!/usr/bin/python
import numpy as np
import sys, os 

def main(argv):
	res = np.zeros([3,3])
	res[0] = calculoPorc("neutral", argv[1], argv[2])
	res[1] = calculoPorc("sad", argv[1], argv[2])
	res[2] = calculoPorc("happy",argv[1], argv[2])
	print(res)
def calculoPorc(emoc, modelDef, modelPre):
	script = "../scripts/execClass.sh " 
	inFile = " ../testImgs3Emoc/"+emoc+"/ "
	outFile = " ../testImgs3Emoc/salida "
	os.system(script+inFile+outFile+" "+modelDef+" "+modelPre)
	matrizRes  = np.load('./salida.npy')
	vecRes = map(np.argmax, matrizRes)
	porc = reduce(miSum, vecRes, [0,0,0])
	return map(lambda x: x*100/len(vecRes), porc)

def miSum(arr, someI):
	arr[someI] = arr[someI] + 1
	return arr

if __name__ == "__main__":
	main(sys.argv)



	
