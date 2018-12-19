####################################
##### AFM Module V 1.0 #####
####################################

#~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage.filters import gaussian_filter1d as gaussFilter
import shlex
import statistics as st
import matplotlib.pyplot as pl
import os
from os.path import join
from numba import jit
import pandas as pd
from collections import OrderedDict
from scipy.special import gamma
from scipy.integrate import quad
from subprocess import check_output as filRun
from os.path import join
from io import StringIO
import pickle
#~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~

'''~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~'''

#~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~
'''Mittag-Leffler Functions'''
#~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~

#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~
'''
Auxilliary Functions
'''	
def Kfunc(_chi, _alpha, _beta, _z):
	
	res  = 	1/(_alpha*np.pi)
	res *= 	_chi**((1-_beta)/_alpha)
	res *=	np.exp(- _chi**(1/_alpha))
	res *=	_chi*np.sin(np.pi*(1-_beta)) - _z*np.sin(np.pi*(1-_beta+_alpha))
	res /= (_chi**2) - 2*_chi*_z*np.cos(_alpha*np.pi) + _z**2

	return res

def Pfunc(_phi, _alpha, _beta, _eps, _z):
	
	omega = _phi*(1 + (1-_beta)/_alpha ) + (_eps**(1/_alpha))*np.sin(_phi/_alpha)
	res =	(1/(2*_alpha*np.pi))*(_eps**(1+(1-_beta)/_alpha))
	res *=	np.exp((_eps**(1/_alpha))*np.cos(_phi/_alpha))
	res *=	np.cos(omega) + 1j*np.sin(omega)
	res /=	_eps*np.exp(1j*_phi) - _z

	return res
	
#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~	

#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~
'''
Main Function
'''	
def mlfMain(alpha, beta, z):
	
	#~ rho = 1e-10
	rho = np.finfo(float).eps
	
#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~	
	
	if 1 < alpha:
		
		k0 = int(np.floor(alpha)+1) #was ceiling in original code
		E_ab = 0.0
		for k in range(k0-1):
			
			_zMod = (z**(1/k0))*np.exp(1j*2*np.pi*k/k0)
			
			E_ab += mlfMain( alpha/k0, beta, _zMod )
			
		return E_ab/k0

#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~	
		
	elif z == 0:
		return 1/gamma(beta)
		
	#~ elif abs(z) < 1e-6: # is this version better?
		#~ return 1/gamma(beta)

#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~	

	elif abs(z) < 1:
		
		k0 = int( max( [np.ceil( (1-beta)/alpha ), 
						np.ceil( np.log( rho*(1 - abs(z))) / np.log(abs(z))) ] ) )
		
		E_ab = 0.0
		for k in range(k0):
			E_ab += (z**k)/gamma(beta + alpha*k)
		
		return E_ab
		
#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~			
		
	elif abs(z) > np.floor(10 + 5*alpha):
		
		k0 = int(np.floor( -np.log(rho) / np.log( abs(z) ) ) )
		
		sumThang = 0.0
		for k in range(1, k0):
			sumThang += (z**(-k))/gamma(beta - alpha*k)
		
		if abs(np.angle(z)) < alpha*np.pi/4 + 1/2*min([np.pi, alpha*np.pi]):
			
			E_ab = (1/alpha)*(z**( (1-beta)/alpha ) )*np.exp(z**(1/alpha) ) - sumThang	
			
		else:
			
			E_ab = - sumThang
			
		return E_ab
		
#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~#~-~

	else:
		
		if beta >= 0:
			chi0 = max( [1, 2*abs(z), (-np.log(np.pi*rho/6))**alpha] )
		elif beta < 0:
			chi0 = max( [(abs(beta) + 1)**alpha, 2*abs(z), (-2*np.log( np.pi*rho/ ( 6 * (abs(beta)+2) * ( (2*abs(beta))**(abs(beta)) ) ) ) )**alpha ] ) 
		
		if abs(np.angle(z)) > alpha*np.pi:
			
			if beta <= 1:
				E_ab, err = quad(Kfunc, 0, chi0, args = (alpha, beta, z) )
			else:
				E_ab, err = quad(Kfunc, 1, chi0, args = (alpha, beta, z) ) + quad(Pfunc, -alpha*np.pi, alpha*np.pi, args = (alpha, beta, 1, z) )
		
		elif abs(np.angle(z)) < alpha*np.pi:
			
			if beta <= 1:
				E_ab, err = quad(Kfunc, 0, chi0, args = (alpha, beta, z) ) + (1/alpha)*(z**((1-beta)/alpha))*np.exp(z**(1/alpha))
			else:
				E_ab, err = quad(Kfunc, abs(z)/2, chi0, args = (alpha, beta, z) ) + quad(Pfunc, -alpha*np.pi, alpha*np.pi, args = (alpha, beta, abs(z)/2, z)) + (1/alpha)*(z**((1-beta)/alpha))*np.exp(z**(1/alpha))
			
		else:
			E_ab, err = quad(Kfunc, (abs(z)+1)/2, chi0, args = (alpha, beta, z)) + quad(Pfunc, -alpha*np.pi, alpha*np.pi, args = (alpha, beta, (abs(z)+1)/2, z) )
		
		return E_ab
		
def mlfPython(times, alpha, tau):
	
	print('Function Called')
	
	mlfOut = np.zeros_like(times)
	for i,t in enumerate(times):
		mlfOut[i] = mlfMain(alpha, 1, -(t/tau)**alpha)
		
	return mlfOut

def MLfunc(_x, _alpha, deriv = False, workingDir = r'C:\Projects\11_Louis\AFM_Module'):
	
	_xPath = join(workingDir, 'xDataMLF.csv')

	np.savetxt(_xPath, _x, delimiter=',')

	output = StringIO(filRun([join(workingDir, 'MLfunc.exe'), str(_alpha), str(len(_x)), _xPath]).decode('utf-8'))

	outputParsed = pd.read_csv(output, delim_whitespace=True, engine = 'c', na_values='Infinity', float_precision = 'round_trip', dtype=np.float64)
	
	if deriv:
		return -outputParsed['Edash(x)']
		
	else:
		return outputParsed['E(x)']
		
def MLfuncCOMPARE(_x, _alpha, deriv = False, workingDir = r'C:\Projects\11_Louis\AFM_Module'):
	
	_xPath = join(workingDir, 'xDataMLF_COMPARE.csv')

	np.savetxt(_xPath, _x, delimiter=',')

	output = StringIO(filRun([join(workingDir, 'MLfuncCOMPARE.exe'), str(_alpha), str(len(_x)), _xPath]).decode('utf-8'))

	outputParsed = pd.read_csv(output, delim_whitespace=True, engine = 'c', na_values='Infinity', float_precision = 'round_trip', dtype=np.float64)
	
	if deriv:
		return -outputParsed['Edash(x)']
		
	else:
		return outputParsed['E(x)']

#~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~
'''Some Hertz Fitting Functions'''
#~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~

def logistic(x, x0, k, Fs = 5000):
	contactDuration = 1*(10**(-10.8)) #seconds
	k = 1/(Fs*contactDuration) 
	return 1 / (1 + np.exp(-k*(-x+x0)))

def hertzInit(data, R, nu = 0.5):
	
	"""To estimate YM and Contact Point"""
	
	#initialise constants
	prefactor = (4*np.sqrt(R))/(3*(1-(nu**2)))
	lenApproach = len( data.data[0] )
	
	#just take last 98% of curve #BUT CHECK FOR NEGATIVITY!
	if len(data.Aheight[data.Aheight<0]) > 0: #check for any -ve values
		
		hifi = np.empty_like(data.Aheight)
		
		for i,v in enumerate(data.Aheight):
			if v >= 0:
				hifi[i] = data.Aheight[i]**(3/2)
			elif v < 0:
				hifi[i] = -((-data.Aheight[i])**(3/2))	
	
	else:
		hifi = data.Aheight**(3/2)
	
	hifi_end = hifi[int(0.98*lenApproach):lenApproach]
	f_end = data.Aforce[int(0.98*lenApproach):lenApproach]
	
	#fit line
	polyz = np.polyfit( hifi_end, f_end, 1)
	
	#generate line
	y = np.zeros( len(hifi) )
	for i,v in enumerate(polyz):
		y += ( v*(hifi**(len(polyz)-i-1)) )
		
	#~ #plot in hifi/force space for diagnostic			
	#~ pl.plot(hifi, data.Aforce)
	#~ pl.plot(hifi, y)
	#~ axInstant = pl.gca()
	#~ axInstant.set_ylim(bottom=-1e-8, top=np.max(f_end)*1.2)
	#~ pl.show()
	
	#Young's Mod	
	YM = -polyz[0]/prefactor
	
	#Contact point (estimated by following line down to y axis and converting to h, no hifi
	cEl = np.argmax( y>0 )
	cpInit = data.Aheight[cEl] 
	
	#~ pl.plot(hifi,f)
	#~ pl.plot(hifi,y, '--')
	#~ pl.show()
		
	"""To estimate offset and tilt"""
	#anything that happens quicker than smoothTimeScale gets smoothed
	smoothTS = 0.1
	Fc = 1/smoothTS
	Fs = 5000 #sample rate
	Fc_halfpower= Fc/np.sqrt(2*np.log(2))
	sig_kernel = Fs/(2*np.pi*Fc_halfpower)
	
	h_start = gaussFilter( data.data[0][0:int(0.7*lenApproach), data.index_hi], sig_kernel )
	f_start = gaussFilter( data.data[0][0:int(0.7*lenApproach), data.index_vd], sig_kernel )
	
	#tilt
	df_dh = np.gradient(f_start, h_start, edge_order=2)
	m = -st.mean(df_dh)
	
	#offset
	b = st.mean(f_start)
			
	return [YM, cpInit, m, b]

def hertzFunk(inz, height, R, nu = 0.5, contactFunk="heaviside"):
	
	prefactor = (4*np.sqrt(R))/(3*(1-(nu**2)))
	
	z = height - inz[1]
	zTrans = np.empty_like(z)
	for i,v in enumerate(z):
		if v >= 0:
			zTrans[i] = z[i]**(3/2)
		elif v < 0:
			zTrans[i] = ((-z[i])**(3/2))
			
	if contactFunk=="heaviside":
		#~ heaviside = np.piecewise(height, [height < 0.0, height >= 0.0], [0, 1]) #why doesn't this work?
		heaviside = np.piecewise(height, [height > inz[1], height <= inz[1]], [0, 1])
		return prefactor*inz[0]*zTrans*heaviside + inz[2]*height + inz[3]
	
	elif contactFunk=="logistic":
		logi = logistic(height, inz[1], 1e-3)
		return prefactor*inz[0]*zTrans*logi + inz[2]*height + inz[3]

def hertzCost(inz, height, R, nu, f, contactFunkIn):
	return f - hertzFunk(inz, height, R, nu=nu, contactFunk=contactFunkIn)

def hertzFit(data, R, nu = 0.5, smoothTS=0.0, contactFunkIn="heaviside"):
	
	#get initial guesses for least squares
	inz0 = hertzInit(data, R, nu) #Youngs Modulus [Pa], Contact Point [m], Force Offset [N], Force Tilt [N/m]
	
	if smoothTS==0.0:
		hSmooth = data.data[0][:,data.index_hi]
		fSmooth = data.data[0][:,data.index_vd]
	elif smoothTS>0.0:
		#anything that happens quicker than smoothTimeScale gets smoothed
		Fc = 1/smoothTS
		Fs = 5000 #sample rate
		Fc_halfpower= Fc/np.sqrt(2*np.log(2))
		sig_kernel = Fs/(2*np.pi*Fc_halfpower)
		
		hSmooth = gaussFilter( data.data[0][:,data.index_hi] , sig_kernel )
		fSmooth = gaussFilter( data.data[0][:,data.index_vd] , sig_kernel )
	
	#distance from contact point initial condition allowed:
	alfie = 0.05
	if inz0[1]<0:
		alfalfa = -alfie #e.g.0.2 would yield 0.8*cp0 for lower bound and 1.2*cp0 for upper bound
	else:
		alfalfa = alfie #e.g.0.2 would yield 0.8*cp0 for lower bound and 1.2*cp0 for upper bound
		
	loBounds = [0, (1-alfalfa)*inz0[1], -5, -5]
	hiBounds = [1e10, (1+alfalfa)*inz0[1], 5, 5]
		
	for i in range(len(inz0)):
		if inz0[i]<loBounds[i]:
			inz0[i] = loBounds[i]
		elif inz0[i]>hiBounds[i]:
			inz0[i] = hiBounds[i]
	
	#fit
	theFit = least_squares(hertzCost, inz0, args=(hSmooth, R, nu, fSmooth, contactFunkIn), verbose=0, xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=10000, method='trf', loss="arctan", bounds=(loBounds, hiBounds))
			
	return theFit['x']
	
######################################################################## Fitting to transformed variables

#~ def hertzFunkTrans(inz, hifi, R, nu = 0.5, contactFunk="heaviside"):
	
	#~ prefactor = (4*np.sqrt(R))/(3*(1-(nu**2)))
	
	#~ zTrans = hifi - inz[1]
			
	#~ if contactFunk=="heaviside":
		#~ heaviside = np.piecewise(height, [height > 0.0, height <= 0.0], [0, 1])
		#~ heaviside = np.piecewise(hifi, [hifi > inz[1], hifi <= inz[1]], [0, 1])
		#~ return prefactor*inz[0]*zTrans*heaviside
	
	#~ elif contactFunk=="logistic":
		#~ logi = logistic(hifi, inz[1], 1e-3)
		#~ return prefactor*inz[0]*zTrans*logi

#~ def hertzCostTrans(inz, hifi, R, nu, f, contactFunkIn):
	#~ return f - hertzFunkTrans(inz, hifi, R, nu=nu, contactFunk=contactFunkIn)
	
#~ def hertzFitTransformed(data, R, nu = 0.5, smoothTS=0.0, contactFunkIn="heaviside"):
	
	#~ """To estimate YM and Contact Point"""	
	#~ #initialise constants
	#~ prefactor = (4*np.sqrt(R))/(3*(1-(nu**2)))
	#~ lenApproach = len( data.data[0] )
	
	#~ #just take last 98% of curve #BUT CHECK FOR NEGATIVITY!
	#~ if len(data.Aheight[data.Aheight<0]) > 0: #check for any -ve values
		
		#~ hifi = np.empty_like(data.Aheight)
		
		#~ for i,v in enumerate(data.Aheight):
			#~ if v >= 0:
				#~ hifi[i] = data.Aheight[i]**(3/2)
			#~ elif v < 0:
				#~ hifi[i] = -(-data.Aheight[i])**(3/2)	
	
	#~ else:
		#~ hifi = data.Aheight**(3/2)
	
	#~ hifi_end = hifi[int(0.98*lenApproach):lenApproach]
	#~ f_end = data.Aforce[int(0.98*lenApproach):lenApproach]
	
	#~ #fit line
	#~ polyz = np.polyfit( hifi_end, f_end, 1)
	
	#~ #generate line
	#~ y = np.zeros( len(hifi) )
	#~ for i,v in enumerate(polyz):
		#~ y += ( v*(hifi**(len(polyz)-i-1)) )
		
	#~ #plot in hifi/force space for diagnostic			
	#~ pl.plot(hifi, data.Aforce)
	#~ pl.plot(hifi, y)
	#~ axInstant = pl.gca()
	#~ axInstant.set_ylim(bottom=-1e-8, top=np.max(f_end)*1.2)
	#~ pl.show()
	
	#~ #Young's Mod	
	#~ YM0 = -polyz[0]/prefactor
	
	#~ #Contact point (estimated by following line down to y axis and converting to h, no hifi
	#~ cEl = np.argmax( y>0 )
	#~ cp0 = hifi[cEl] 
	
	#~ #distance from contact point initial condition allowed:
	#~ if cp0<0:
		#~ alfalfa = -0.2 #e.g.0.2 would yield 0.8*cp0 for lower bound and 1.2*cp0 for upper bound
	#~ else:
		#~ alfalfa = 0.2 #e.g.0.2 would yield 0.8*cp0 for lower bound and 1.2*cp0 for upper bound
	
	#~ #fit
	#~ theFit = least_squares(hertzCostTrans, [YM0, cp0], args=(hifi, R, nu, data.Aforce, contactFunkIn), verbose=0, xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=10000, method='trf', loss="arctan", bounds=( [0, (1-alfalfa)*cp0], [1e10, (1+alfalfa)*cp0] ) )
				
	#~ return [ theFit['x'][0]**(3/2), theFit['x'][1]**(2/3) ]
 
#~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~
'''Main Class with Data Reading Functions'''
#~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~

class AFMdata:
	
	def __init__(self, filDir, concatenate=False):
		
		#initialise Empty List for finding Different Sections
		zerzz = []
		
		# Open File for Reading, Extract Lines and Close.
		with open(filDir, 'r') as f:
			for i,line in enumerate(f):
				# Extract Numerical Data 
				if line.startswith("# units: "):            
					zerzz.append(i + 1)    
				elif line=="\n":
					zerzz.append(i)
				elif line.startswith("# fancyNames:"):
					fancy_names_all = line
				elif line.startswith("# data-description.modification-software:"):
					verLine = line
				
			#append last line
			zerzz.append(i+1)
			
		dataBlock = pd.read_csv(filDir, delim_whitespace = True, comment='#', header=None, engine = 'c', float_precision = 'round_trip', dtype=np.float64).values
		
		#surely not the best but it works
		blockLengths = [0]*int(len(zerzz)/2)
		blockMod = [0]*int(len(zerzz)/2)
		blockCounter = 0
		for q in range(int(len(zerzz)/2)):
			blockCounter += blockLengths[q-1]
			blockLengths[q] = zerzz[1::2][q] - zerzz[0::2][q]
			blockMod[q] = blockLengths[q] + blockCounter - (q+1)
		
		blockMod = [0] + blockMod
		
		#~ print(blockMod)
		
		self.data = [0]*len(blockLengths)
		for q in range( len(blockLengths) ):
			self.data[q] = dataBlock[ blockMod[q] : blockMod[q+1] ]
				
		#different names for different versions
		verNum = verLine.split()[2]
			
		#organise fancy names and store to class
		self.fancyNames = shlex.split( fancy_names_all )
		
		del self.fancyNames[0] , self.fancyNames[0] 
		
		#get correct column numbers
		self.index_vd = self.fancyNames.index("Vertical Deflection") #force is always the same
		
		#might just be elastic
		try:
			self.index_ti = self.fancyNames.index("Series Time")
		except ValueError:
			pass
		
		#corrected height name depends on version
		if verNum == "spm-5.0.69":
			self.index_hi = self.fancyNames.index("Tip-Sample Separation")
		elif verNum == "spm-6.1.22":
			self.index_hi = self.fancyNames.index("Vertical Tip Position")
			
		#approach height and force
		self.Aheight = self.data[0][:, self.index_hi]
		self.Aforce = self.data[0][:, self.index_vd]
						
	def fixedThreshold(self, fThreshold, allSections = False, sampleRate = 5000, delRetract = True, smoothTimeScale = 0.001, tDependent = True):
		
		# IF JUST WANT APPROACH THEN USE AS NORMAL, IF WANT ALL SECTIONS 
		# COMBINED AS OUTPUT THEN SET "allSections = True"
		
		if allSections:
			#combine all sections but retract
			del self.data[-1]
			self.dataThresh = [np.vstack(self.data)]
		else:
			#just same
			self.dataThresh = self.data
		
		#init
		lenSegment = len( self.dataThresh[0] ) # length of approach/all sections
		
		nu_start = np.argmax( self.dataThresh[0][:,self.index_vd]>fThreshold ) # first element where fixed force threshold is exceeded
		
		#get array of forces
		self.forceUS = self.dataThresh[0][nu_start:lenSegment,self.index_vd] - min(self.dataThresh[0][nu_start:lenSegment,self.index_vd])

		#get t
		if tDependent:
			time_cropped = self.dataThresh[0][nu_start:lenSegment, self.index_ti]
			self.t = time_cropped - min(time_cropped)
					
		#crop height accordingly
		height_cropped = self.dataThresh[0][:,self.index_hi][nu_start:lenSegment]
		height_cropped *= -1
		self.heightUS = height_cropped - min(height_cropped)
		
		# smooth concatenated data 
		# anything that happens quicker than smoothTimeScale gets smoothed
		Fc = 1/smoothTimeScale
		
		Fs = sampleRate #sample rate
		
		Fc_halfpower= Fc/np.sqrt(2*np.log(2))
		
		sig_kernel = Fs/(2*np.pi*Fc_halfpower)
		
		#height smooth
		self.heightS = gaussFilter(self.heightUS, sig_kernel)
		#force smooth
		self.forceS = gaussFilter(self.forceUS, sig_kernel)
		
		#numerically differentiate load
		if tDependent:
			self.dl_dt = np.gradient(self.forceS, self.t, edge_order=2)
		
		#get hifi
		self.hifi = self.heightS**(3/2)
		
		if tDependent:	
			self.dhifi_dt = np.gradient(self.hifi, self.t, edge_order=2)
		
	def hertzThreshold(self, elasticInfo, allSections = False, sampleRate = 5000, delRetract = True, smoothTimeScale = 0.001, tDependent = True):
		
		# IF JUST WANT APPROACH THEN USE AS NORMAL, IF WANT ALL SECTIONS 
		# COMBINED AS OUTPUT THEN SET "allSections = True"
		
		if allSections:
			#combine all sections but retract
			del self.data[-1]
			self.dataThresh = [np.vstack(self.data)]
		else:
			#just same
			self.dataThresh = self.data
		
		#init
		lenSegment = len( self.dataThresh[0] ) # length of approach/all sections
		
		nu_start = np.argmax( self.dataThresh[0][:,self.index_hi]<elasticInfo[1] ) #element closest to contact point found by hertz fit
		
		#get array of forces
		self.forceUS = self.dataThresh[0][nu_start:lenSegment,self.index_vd] - min(self.dataThresh[0][nu_start:lenSegment,self.index_vd])

		#get t
		if tDependent:
			time_cropped = self.dataThresh[0][nu_start:lenSegment, self.index_ti]
			self.t = time_cropped - min(time_cropped)
					
		#crop height accordingly
		height_cropped = self.dataThresh[0][:,self.index_hi][nu_start:lenSegment]
		self.heightUS = -1*(height_cropped - elasticInfo[1])

		# smooth concatenated data 
		# anything that happens quicker than smoothTimeScale gets smoothed
		Fc = 1/smoothTimeScale
		
		Fs = sampleRate #sample rate
		
		Fc_halfpower= Fc/np.sqrt(2*np.log(2))
		
		sig_kernel = Fs/(2*np.pi*Fc_halfpower)
		
		#height smooth
		self.heightS = gaussFilter(self.heightUS, sig_kernel)
		#force smooth
		self.forceS = gaussFilter(self.forceUS, sig_kernel)
		
		#numerically differentiate load
		if tDependent:
			self.dl_dt = np.gradient(self.forceS, self.t, edge_order=2)
		
		#get hifi
		self.hifi = self.heightS**(3/2)
		
		if tDependent:	
			self.dhifi_dt = np.gradient(self.hifi, self.t, edge_order=2)
				
	def fdfHertz(self, hertzOut):
		#returns final indentation depth and force, with contact height found by Hertz fit
		return np.asarray([ (hertzOut[1] - self.Aheight[-1]), self.Aforce[-1], hertzOut[0] ])
		
	def dAndYM_Hertz(self, hertzOut):
		#returns final indentation depth and force, with contact height found by Hertz fit
		return (hertzOut[1] - self.Aheight[-1]), hertzOut[0]
		
	def absContactHeight(self, fThreshold):

		#starting element
		nu_start = np.argmax( self.data[0][:,self.index_vd]>fThreshold ) # first element where fixed force threshold is exceeded
		
		#get absolute height value
		return self.data[0][:,self.index_hi][nu_start]
		
#~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~
'''CREEP Viscoelastic Fitting for Spherical/Paraboloidal Indenter'''
#~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~

# Creep Moduli
def J_SLS(inz, t):
	
	return inz[0] - inz[1]*np.exp(-t/inz[2])
	
def J_SLS2(inz, t):
	
	return inz[0] - inz[1]*np.exp(-t/inz[2]) - inz[3]*np.exp(-t/inz[4])
	
def J_Burgers(inz, t):
	
	return inz[0] - inz[1]*np.exp(-t/inz[2]) + inz[3]*t	
	
def J_FractZener(inz,t):
	
	ans = (1/2)*(inz[0] + inz[1]*(1 - MLfunc(-(t/inz[2])**inz[3], inz[3]))) #1/mu1, 1/mu2, tau, nu
	
	return ans

# Main CREEP Least Squares Funks	
def Jfunk(inz, t, dl_dt, R, model2use):

	Jmodels = {'SLS':J_SLS, 
				'SLS2':J_SLS2, 
				'burgers':J_Burgers,
				'FractZener':J_FractZener}
				
	ans = ((3/8)/(R**(1/2)))*np.multiply( np.convolve( Jmodels[model2use](inz,t) , dl_dt , mode='full')[0:len(t)] , np.gradient(t) )
				
	#~ print(ans)
   
	#~ return ((3/8)/(R**(1/2)))*np.multiply( np.convolve( Jmodels[model2use](inz,t) , dl_dt , mode='full')[0:len(t)] , np.gradient(t) )
	return ans

# Cost Funkyshun   
def Jresidual(paramz, t, dl_dt, R, hifi, model2use):
	
    return Jfunk(paramz, t, dl_dt, R, model2use) - hifi
    
# Fit Funky
def creepFit(data, R, model2use='SLS', hertzian = False):

	Jparams0 = {'SLS':[1.0,0.5,1.0], 
				'SLS2':[1.0,0.5,1.0,1.0,1.0], 
				'burgers':[1.0,0.5,1.0,1.0],
				'FractZener':[1e-1, 1e-1, 1e3, 0.5]}
				
	if model2use=='FractZener':			
	
		return least_squares(Jresidual, Jparams0[model2use], args=(data.t, data.dl_dt, R, data.hifi, model2use), bounds = ([1e-7, 1e-7, 1e-9, 1e-9],[1.0, 1.0, np.inf, 1.0]), verbose=2, ftol=1e-15, gtol=1e-15, method='trf')
	
	else:
	
		return least_squares(Jresidual, Jparams0[model2use], args=(data.t, data.dl_dt, R, data.hifi, model2use), verbose=2, ftol=1e-15, gtol=1e-15, method='trf')
	
#Get Parameters from scipy fit output dictionary
def creep_slsGetParams(fit_out):

	Ke = 1/fit_out['x'][0]
	
	Km = ( fit_out['x'][1]*Ke*Ke ) / ( 1 - Ke*fit_out['x'][1] )
	
	Cm = (Ke*Km*fit_out['x'][2]) / (Ke + Km)
	
	return (Ke, Km, Cm)
	
# For Plotting	
def JtestOut(inz, data, R, model2use='SLS'):
	
	Jmodels = {'SLS':J_SLS, 'SLS2':J_SLS2, 'burgers':J_Burgers}
	
	t = data.t
	dl_dt = data.dl_dt
	convo = ((3/8)/(R**(1/2)))*np.multiply( np.convolve( Jmodels[model2use](inz,t) , dl_dt , mode='full')[0:len(t)] , np.gradient(t) )
	
	out = (convo**(2))**(1/3)
	
	return out

#~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~
'''STRLX Viscoelastic Fitting for Spherical/Paraboloidal Indenter'''
#~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~

# Relaxation Moduli
def G_SLS(inz,t):
	
	return inz[0] + inz[1]*np.exp(-t/inz[2])
	
def G_SLS2(inz,t):
	
	return inz[0] + inz[1]*np.exp(-t/inz[2]) + inz[3]*np.exp(-t/inz[4])
	
def G_Burgers(inz,t):
	
	return inz[0]*np.exp(-t/inz[1]) + inz[2]*np.exp(-t/inz[3])
	
def G_PowerLaw(inz, t):
	
	return inz[0] + ( (inz[1] - inz[0]) / ( (1 + t/inz[2])**inz[3] ) )
	
def G_SpringPot(inz,t):
	
	return inz[0]*(t**(-inz[1]))/gamma(1-inz[1])
	
def G_FractMaxwell(inz,t,compare=False):
	
	if compare:
	
		return 2*inz[0]*MLfunc(-(t/inz[2])**inz[1], inz[1])
	
	else:
	
		return 2*inz[0]*MLfunc(-(t/inz[2])**inz[1], inz[1])
	
def G_FractZener(inz,t, compare=False):
	
	if compare:
	
		return 2*inz[0]*( 1 + inz[1]*MLfuncCOMPARE(-(t/inz[2])**inz[3], inz[3]) ) #mu, r, tau, nu
	
	else:
		
		return 2*inz[0]*( 1 + inz[1]*MLfunc(-(t/inz[2])**inz[3], inz[3]) ) #mu, r, tau, nu
	
# Main STRLX Least Squares Funks
def Gfunk(inz, data, R, model2use, compare = False):

	Gmodels = {	'SLS':G_SLS, 
				'SLS2':G_SLS2, 
				'burgers':G_Burgers,
				'PowerLaw':G_PowerLaw,
				'SpringPot':G_SpringPot,
				'FractMaxwell':G_FractMaxwell,
				'FractZener':G_FractZener}
	
	if model2use == 'FractMaxell' or model2use == 'FractZener' and compare:
		
		return ((8*(R**(1/2)))/3)*np.multiply(np.convolve(Gmodels[model2use](inz,data.t, compare=True), data.dhifi_dt, mode='full')[0:len(data.t)], np.gradient(data.t) )
		
	else:
		
		return ((8*(R**(1/2)))/3)*np.multiply(np.convolve(Gmodels[model2use](inz,data.t), data.dhifi_dt, mode='full')[0:len(data.t)], np.gradient(data.t) )

# Cost Funkyshun
def Gresidual(paramz, data, R, model2use):
	
    return Gfunk(paramz, data, R, model2use) - data.forceS
    
def strlxFit(data, R, model2use='SLS', params0 = 'none'):
	
	if params0=='none':
	
		Gparams0 = { 	'SLS':[1000.0, 1000.0, 1000.0], 
						'SLS2':[1000.0, 1000.0, 1000.0, 1000.0, 1000.0], 
						'burgers':[1000.0, 1000.0, 1000.0, 1000.0], 
						'PowerLaw':[1000.0, 1000.0, 5.0, 0.5],
						'SpringPot':[1.0, 0.5],
						'FractMaxwell':[20000.0, 0.17, 0.25],
						'FractZener':[20000.0, 1.0, 0.25, 0.25] }
						
	else:
		
		Gparams0 = params0
	
	try:
	
		boundsLo = [0]*len(Gparams0[model2use])
		boundsHi = [1e10]*len(Gparams0[model2use])
			
		if model2use=='FractMaxwell':
			return least_squares(Gresidual, Gparams0[model2use], args=(data, R,  model2use), bounds = ([1e-9,1e-9,1e-9],[np.inf,1,100]), verbose=2, ftol=1e-12, method='trf')
		else:
			return least_squares(Gresidual, Gparams0[model2use], args=(data, R,  model2use), verbose=2, ftol=1e-15, gtol=1e-15, bounds=( boundsLo, boundsHi ), method='trf')

	except TypeError:
		
		print('Initial Conditions specified.')
		print(Gparams0)
		
		boundsLo = [0]*len(Gparams0)
		boundsHi = [1e10]*len(Gparams0)
			
		if model2use=='FractMaxwell':
			return least_squares(Gresidual, Gparams0, args=(data, R,  model2use), bounds = ([1e-9,1e-9,1e-9],[np.inf,1,100]), verbose=2, ftol=1e-12, method='trf')
		if model2use=='FractZener':
			return least_squares(Gresidual, Gparams0, args=(data, R,  model2use), bounds = ([1e-9,1e-9,1e-9],[np.inf,np.inf,np.inf,0.99999999999]), verbose=2, ftol=1e-14, method='trf')
		else:
			return least_squares(Gresidual, Gparams0, args=(data, R,  model2use), verbose=2, ftol=1e-15, gtol=1e-15, bounds=( boundsLo, boundsHi ), method='trf')
		
#Get Parameters from scipy fit output dictionary
def strlx_slsGetParams(fit_out):
	
	Ke = fit_out['x'][0]
	
	Km = fit_out['x'][1]
	
	Cm = fit_out['x'][2]*Km
	
	return (Ke, Km, Cm)
	
#~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~
'''Analysis of Pickle Files - Collection of Functions'''
#~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~

#~ def pickleWalker(folDir, mType = False, tType = False, wType = False):
	
	#~ #walk through dir
	#~ for dName1, sdList1, fList1 in os.walk(folDir): #Traverse each biological repeat
		#~ for sd in sdList1:
			
			#~ if wallType in sd:
			
				#~ #get full directory for current biological repeat
				#~ current_bio = join(dName1,sd)
				
				#~ for dName2, sdList2, flist2 in os.walk(current_bio):
					#~ for f in flist2:
						
						#~ if f.endswith(mType+'_'+tType+'.pickle'):
							
							#~ #get abs. directory of pickle file
							#~ filDir = join(current_bio,f)
							#~ print(filDir)
							
							#~ with open(filDir, 'rb') as q:
								#~ curData = pickle.load(q)
								
							#~ if not curData['status']==-1 and not curData['status']==0:
							
								#~ paramz.append(curData['x'])
								#~ costzz.append(curData['cost'])
								#~ namezz.append(filDir)
								
							#~ else:
								#~ print('dodgy file', f)	
								
	#~ return paramz, costzz, namezz

def strlxAnalyse(paramz, costzz, namezz, mType = False):
	
	''' Function accepts parameters of various models as lists and sublists,
	converts to array and cleans up time scale outliers and negative parameters.
	If required, also deletes corresponding info from costs and file names for deeper
	analysis.'''

	paramz = np.vstack(paramz)
	
	#sort so time scales are in increasing order e.g. instantaneous, shorter, longer (not necesssary for SLS1)
	if mType == 'BURG':
		for i,v in enumerate(paramz):
			if paramz[i,1] > paramz[i,3]:
				paramz[i,1],paramz[i,3] = paramz[i,3],paramz[i,1]
				paramz[i,0],paramz[i,2] = paramz[i,2],paramz[i,0]
				
	elif mType == 'SLS2':
		for i,v in enumerate(paramz):
			if paramz[i,2] > paramz[i,4]:
				paramz[i,2],paramz[i,4] = paramz[i,4],paramz[i,2]
				paramz[i,3],paramz[i,1] = paramz[i,1],paramz[i,3]
				
	#Now check for obvious outliers in time scale, in this larger analysis that means median*10
	if mType == 'BURG':
		tau1_median = st.median(paramz[:,1])
		tau2_median = st.median(paramz[:,3])
	elif mType == 'SLS1':
		tau_median = st.median(paramz[:,2])
	elif mType == 'SLS2':
		tau1_median = st.median(paramz[:,2])
		tau2_median = st.median(paramz[:,4])
	elif mType == 'FractZener':
		tau_median = st.median(paramz[:,2])
		rMu_median = st.median(paramz[:,1])
	
	#for storing wronguns
	outlier_indices = []
	
	if mType == 'BURG':
		for i,(v1,v2) in enumerate(zip(paramz[:,1], paramz[:,3])):
			if v1>tau1_median*10 or v2>tau2_median*10:
				outlier_indices.append(i) 
	elif mType == 'SLS1':
		for i,v in enumerate(paramz[:,2]):
			if v > tau_median:
				outlier_indices.append(i)
	elif mType == 'SLS2':
		for i,(v1,v2) in enumerate(zip(paramz[:,2], paramz[:,4])):
			if v1>tau1_median*10 or v2>tau2_median*10:
				outlier_indices.append(i)
	elif mType == 'FractZener':
		for i, (v1, v2) in enumerate(zip(paramz[:,2],paramz[:,1])):
			if v1>tau_median*10:
				outlier_indices.append(i)
			if v2>rMu_median*10:
				outlier_indices.append(i) 	 

	paramzTimed = np.delete(paramz, outlier_indices, axis=0)
	costzzTimed = np.delete(costzz, outlier_indices, axis=0)
	namezzTimed = np.delete(namezz, outlier_indices, axis=0)

	#Delete any files with negative parameters	
	outliersNeg = []

	for i1,v1 in enumerate(paramzTimed):
		for i2,v2 in enumerate(v1):
			if v2<0:
				ouliersNeg.append(i)
				
	paramzClean = np.delete(paramzTimed, outliersNeg, axis=0)
	costzzClean = np.delete(costzzTimed, outliersNeg, axis=0)
	namezzClean = np.delete(namezzTimed, outliersNeg, axis=0)
	
	return paramzClean, costzzClean, namezzClean
	
#~ def creepAnalyse(paramz, costzz = [], namezz = [], mType = False):
	
	#~ ''' Function accepts parameters of various models as lists and sublists,
	#~ converts to array and cleans up time scale outliers and negative parameters.
	#~ If required, also deletes corresponding info from costs and file names for deeper
	#~ analysis.'''

	#~ if costzz == []:
		#~ costzz = [0]*len(paramz)
		
	#~ if namezz == []:
		#~ namezz = [0]*len(parmz)
			
	#~ paramz = np.vstack(paramz)
	
	#~ #sort so time scales are in increasing order e.g. instantaneous, shorter, longer (not necesssary for SLS1)
	#~ if mType == 'BURG':
		#~ for i,v in enumerate(paramz):
			#~ if paramz[i,1] > paramz[i,3]:
				#~ paramz[i,1],paramz[i,3] = paramz[i,3],paramz[i,1]
				#~ paramz[i,0],paramz[i,2] = paramz[i,2],paramz[i,0]
				
	#~ if mType == 'SLS2':
		#~ for i,v in enumerate(paramz):
			#~ if paramz[i,2] > paramz[i,4]:
				#~ paramz[i,2],paramz[i,4] = paramz[i,4],paramz[i,2]
				#~ paramz[i,3],paramz[i,1] = paramz[i,1],paramz[i,3]
				
	#~ #Now check for obvious outliers in time scale, in this larger analysis that means median*10
	#~ if mType == 'BURG':
		#~ tau1_median = st.median(paramz[:,1])
		#~ tau2_median = st.median(paramz[:,3])
	#~ if mType == 'SLS1':
		#~ tau_median = st.median(paramz[:,2])
	#~ if mType == 'SLS2':
		#~ tau1_median = st.median(paramz[:,2])
		#~ tau2_median = st.median(paramz[:,4])
	
	#~ #for storing wronguns
	#~ outlier_indices = []
	
	#~ if mType == 'BURG':
		#~ for i,(v1,v2) in enumerate(zip(paramz[:,1], paramz[:,3])):
			#~ if v1>tau1_median*10 or v2>tau2_median*10:
				#~ outlier_indices.append(i) 
	#~ if mType == 'SLS1':
		#~ for i,v in enumerate(paramz[:,2]):
			#~ if v > tau_median:
				#~ outlier_indices.append(i)
	#~ if mType == 'SLS2':
		#~ for i,(v1,v2) in enumerate(zip(paramz[:,2], paramz[:,4])):
			#~ if v1>tau1_median*10 or v2>tau2_median*10:
				#~ outlier_indices.append(i) 

	#~ paramzTimed = np.delete(paramz, outlier_indices, axis=0)
	#~ costzzTimed = np.delete(costzz, outlier_indices, axis=0)
	#~ namezzTimed = np.delete(namezz, outlier_indices, axis=0)

	#~ #Delete any files with negative parameters	
	#~ outliersNeg = []

	#~ for i,v in enumerate(paramzTimed):
		#~ for j,w in enumerate(v):
			#~ if j<0:
				#~ ouliersNeg.append(i)
				
	#~ paramzClean = np.delete(paramzTimed, outliersNeg, axis=0)
	#~ costzzClean = np.delete(costzzTimed, outliersNeg, axis=0)
	#~ namezzClean = np.delete(namezzTimed, outliersNeg, axis=0)
	
	#~ return paramzClean, costzzClean, namezzClean
	
#~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~
'''Fit Evaluation'''
#~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~

def G_SpringPot_COMPARE(inz, t):
	
	return inz[0]*(t**(-inz[1]))/gamma(1-inz[1])

def G_FractMaxwell_COMPARE(inz,t):
	
	return 2*inz[0]*mlfPython(t, inz[1], inz[2]) 

	
def G_FractZener_COMPARE(inz,t):
	
	return 2*inz[0]*( 1 + inz[1]*mlfPython(t, inz[3], inz[2]) ) 
	
def G_FractZenerINITt0(inz):
	
	return 2*inz[0]*( 1 + inz[1]*mlfPython([0], inz[3], inz[2]) ) 
	
def Gfunk_COMPARE(inz, data, R, model2use):

	Gmodels = {	'SLS':G_SLS, 
				'SLS2':G_SLS2, 
				'BURG':G_Burgers,
				'PowerLaw':G_PowerLaw,
				'SpringPot':G_SpringPot_COMPARE,
				'FractMaxwell':G_FractMaxwell_COMPARE,
				'FractZener':G_FractZener_COMPARE }
				
	if model2use == 'SpringPot':
		
		return ((8*(R**(1/2)))/3)*np.multiply(np.convolve(Gmodels[model2use](inz,data.t), data.dhifi_dt, mode='full')[0:len(data.t)], np.gradient(data.t) )[5000:]
		
	else:

		return ((8*(R**(1/2)))/3)*np.multiply(np.convolve(Gmodels[model2use](inz,data.t), data.dhifi_dt, mode='full')[0:len(data.t)], np.gradient(data.t) )
	
def J_FractZener_COMPARE(inz,t):
	
	return (1/2)*(inz[0] + inz[1]*(1 - mlfPython(t, inz[3], inz[2]))) #1/mu1, 1/mu2, tau, nu

def Jfunk_COMPARE(inz, t, dl_dt, R, model2use):

	Jmodels = {'SLS':J_SLS, 
				'SLS2':J_SLS2, 
				'burgers':J_Burgers,
				'FractZener':J_FractZener_COMPARE}
				
	ans = ((3/8)/(R**(1/2)))*np.multiply(np.convolve(Jmodels[model2use](inz,t), dl_dt, mode='full')[0:len(t)], np.gradient(t))
				
	#~ print(ans)
   
	#~ return ((3/8)/(R**(1/2)))*np.multiply( np.convolve( Jmodels[model2use](inz,t) , dl_dt , mode='full')[0:len(t)] , np.gradient(t) )
	return ans
	
def compareFile2Fit(filDir, ext, modelUsed, R, threshold = None, loglog = False, show = False, testType = 'strlx', HertzThreshold = False): 
	
	data = AFMdata(filDir) #get data object
	
	if HertzThreshold:
		
		ElasticInfo = hertzFit(data, R)
		data.hertzThreshold(ElasticInfo, allSections=True)
		
	elif threshold != None:
	
		data.fixedThreshold(threshold, allSections = True, smoothTimeScale = 0.002)
	
	pickleDir = filDir + ext
	
	with open(pickleDir, 'rb') as f:
	    params = pickle.load(f)['x']
	    
	print(params)
	    
	if testType == 'strlx':
	
		fit2 = Gfunk_COMPARE(params, data, R, modelUsed)
		#~ fit2 = Gfunk(params, data, R, modelUsed)
		
	elif testType == 'creep':
		
		fit2 = (Jfunk_COMPARE(params, data.t, data.dl_dt, R, modelUsed)**2)**(1/3)
	
	if loglog:
		
		if testType == 'strlx':
			pl.loglog(data.t, data.forceS)
		elif testType == 'creep':
			pl.loglog(data.t, data.heightS)
			
		pl.loglog(data.t, fit2, '--')

	else:
		
		if testType == 'strlx':
			pl.plot(data.t, data.forceS)
		elif testType == 'creep':
			pl.plot(data.t, data.heightS)
			
		pl.plot(data.t, fit2, '--')
		
	pl.show()

#~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~
'''Plotting - Collection of Functions'''
#~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~

def removeLegendDuplicates(axIn):
	'''Use to remove duplicates in legend entries.
	Generates tuple, correct syntax is:
	
	legVals, legKeys = pumpkin.removeLegendDuplicates( pl.gca() )
	pl.legend( legVals, legKeys, loc = 'best' ) 
	
	Change location to suit needs'''	
	handles, labels = axIn.get_legend_handles_labels()
	by_label =  OrderedDict(zip(labels, handles))
	return (by_label.values(), by_label.keys())

#~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~
'''Force Mapping Functions'''
#~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~

def tsvRead(filDir):
	with open(filDir):
		tsvData = pd.read_csv(filDir, delimiter='\t') # Read data from CSV file
		
		return list(tsvData["Young's Modulus [Pa]"]), list(tsvData["Contact Point [m]"]) # Return Young's Moduli and Contact Points as a tuple of lists

def makeSquare(listIn, fastDim, slowDim):
	
	# 0 indexed sizes
	slowB = slowDim - 1
	fastB = fastDim - 1
	
	aSquare = np.empty((slowDim, fastDim)) # empty array for allocation
	
	aSquare[slowB, 0:fastDim] = listIn[0:fastDim] # assign bottom row
	
	for k in range(1, slowDim):
	
		if (slowB-k)%2!=0:
			aSquare[slowB - k, 0:fastDim] = listIn[k*fastDim:(k+1)*fastDim]
		
		elif (slowB-k)%2==0:
			aSquare[slowB - k, 0:fastDim] = listIn[k*fastDim:(k+1)*fastDim][::-1]
			
	return aSquare

#~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~
'''File Management Scripts'''
#~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~-~##~

def PICKLE_REMOVAL_USE_CAREFULLY(folDir):
	#removes all pickle folders in folder given and its subdirs. USE WITH CAUTION
	for dName, sdList, fList in os.walk(rootDir):
		for f in fList:
			if f.endswith('.pickle'):
				fullDir = join(dName,f)
				os.remove(fullDir)
				
#~ def folDist(folDir, sepNum = 5):
	#~ for dz1, sdz1, fz1 in os.walk(folDir):
		
	


