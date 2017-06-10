#cuenta cuantos archivos hay con cada cantidad y clase de etiquetas.
# para ejecutar en el dir. AULabels.
import numpy as np
import os, csv
rootdir = os.getcwd()
def funAux(index, myItem):
	del myItem[index]
	return(myItem)

def searchAndDelete(name, lines):
	header = lines[0]
	if name in header:
		index = header.index(name)
		newLines = map(lambda x: funAux(index,x), lines)
		return(newLines)
	else: return(lines)
def miFunMap(last, item):
	item[last] = item[last].strip()
	item[last] = item[last].strip('\"')
	return item

for fi in os.listdir(rootdir):
  if fi.endswith("-label.csv"):
        absFile = os.path.join(rootdir, fi)
	#print "archivo:\t"+absFile+"\n"
	with open(absFile) as f:
		print fi
    lines = f.readlines()
		lines = map(lambda x: x.split(','), lines)
		
		lines[:][0] = map(lambda x: x.strip('\''),lines[:][0])
		for name in ['57','58','video_Glasses', 'video_Gender', 'video_HairBangs', 'video_Lighting', 'Social', 'Oclusion', 'Occlusion','video_Lightin','video_Glasse','video_HairBang','Trackerfail','video_Gende']:
			lines = searchAndDelete(name, lines)
		lines[0] = map(lambda x: x.upper(), lines[0])
		last= len(lines[:][1])-1
		lines = map(lambda x: miFunMap(last,x),lines) 
		#print lines[0]
		with open(os.path.join(rootdir,"new-"+fi),'w') as m:
			writer = csv.writer(m)
			writer.writerows(lines)
#		if tot==1:
#			masterHeader = lines[0]
#		if lines[0] != masterHeader:
#			print fi
#			print lines[0]
#		truth = map(lambda x:len(x) == len(lines[0]),lines)
#		redTruth = reduce(lambda x,y: x and y,truth,True)
#		if not(redTruth):
#			print fi	
#			algo = algo + 1
#			print lines[0]
#			print lines[1]
#print algo
			
