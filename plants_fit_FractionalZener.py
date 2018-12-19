import pickle
import os
from os.path import join
import matplotlib.pyplot as pl
import importlib.util
spec = importlib.util.spec_from_file_location("module.name", "V:\\Projects\\11_Louis\\AFM_Module\\AFM_v1_0.py")
pumpkin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pumpkin)

def analyseCreepTHRESH(filDir):

	R = 10e-9
	beeta = 1e-8
	
	print("Processing file:",filDir)
	
	data = pumpkin.AFMdata(filDir) #get data object
	data.fixedThreshold(beeta, allSections = True, smoothTimeScale = 0.002)
	
	fittingsFractZener = pumpkin.creepFit(data, R, model2use='FractZener')
	
	with open(filDir + '_FractZener_creep.pickle', 'wb') as f:
		pickle.dump(fittingsFractZener, f)
	
#~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~ 
'''Initialise and Run'''
#~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~ 

#User Input for folder
rootDir = "V:\\Projects\\11_Louis\\GelsProject\\20170516 FB data 0 suc Day 2"
#~ rootDir = "V:\\Projects\\11_Louis\\GelsProject\\20170524 FB data 0 suc Day 1"

for dName1, sdList1, fList1 in os.walk(rootDir):
	for f in fList1:
		if f.endswith('.txt'):
			_filDir = join(dName1, f)
			analyseCreepTHRESH(_filDir)
					
