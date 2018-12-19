import pickle
import os
from os.path import join
import AFM_v1_0 as pumpkin

def analyseStrlx(fName,fDir,DMnum):

	if DMnum=="DM33" or DMnum=="DM41":
		beeta = 10**(-7)
		
	elif DMnum=="DM41 m33_50":
		beeta = 10**(-7.2)
		
	elif DMnum=="DM50" or DMnum=="DM50 m41_60" or DMnum=="DM50 m33_70":
		beeta = 10**(-7.75)
		
	elif DMnum=="DM40 Random 2um":
		beeta = 10**(-7.5)
		
	elif DMnum=="DM40 Block":
		beeta = 10**(-7.3)

	R = 25e-6
	
	filDir = join(fDir,fName)
	
	print("Processing file:",filDir)
	
	data = pumpkin.AFMdata(filDir) #get data object

	data.fixedThreshold(beeta, allSections = True, smoothTimeScale = 0.002) #threshold by force and concatenate

	fittingsSLS2 = pumpkin.strlxFit(data, R, model2use='SLS2')
	# with open(fDir+'\\'+fName+'_SLS2_strlx.pickle', 'wb') as f:
	# 	pickle.dump(fittingsSLS2, f)

#~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~
'''Initialise and Run'''
#~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~

#User Input for folder
rootDir = "V:\\Projects\\11_Louis\\GelsProject\\20170214_allvisco_reprocessedandthreshed"

for dName1, sdList1, fList1 in os.walk(rootDir): #Traverse each DM
	for sd1 in sdList1:

		#get full directory for current DM folder
		current_DM = join(dName1,sd1)
		
		if not sd1=="DM40 Block":

			for dName2, sdList2, fList2 in os.walk(current_DM): #Traverse each biological repeat
				for sd2 in sdList2:
					
					#get full directory for current biological repeat
					current_bio = join(dName2,sd2)
					
					for dName3, sdList3, flist3 in os.walk(current_bio):
						for f in flist3:
							if f.startswith('strlx') and f.endswith('.txt'):
								analyseStrlx(f,current_bio,sd1)
