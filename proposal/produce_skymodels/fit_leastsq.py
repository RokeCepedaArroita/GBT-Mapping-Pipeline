import numpy as np
from scipy.optimize import curve_fit

# fit_leastsq.py: uses a library of fitting functions and scipy's curve_fit to return fit
# parameters and parameter errors by using a least squares minimisation



def fit_leastsq(x, y, modelname, initial_guess=None, errors=None):


	# x, y: row vectors with x and y data
	# modelname: name of the model: poly1origin, gaussian, sinfit, cosfit, exponential, lorentzian, cosecant, poly0-9.
	# initial_guess: optional row vector containing the initial variables, important with noisy data.
	# errors: optional row vector of y data errors



	if errors is not None:

		absolute_sigma = True
	else:

		absolute_sigma = False



	## Reduced Chi-Squared Calculator

	def red_chi2(ydata,fit,p,errors=None):

		if errors is not None:

			chi_sq = np.sum(np.divide(np.power(ydata-fit,2),np.power(errors,2)))/(len(ydata)-len(p))
		else:

			chi_sq = np.sum(np.power(ydata-fit,2))/(len(ydata)-len(p))


		return chi_sq



	## Default MATLAB functions



	#--------------poly1origin------------------#

	def poly1origin(x, p0):
		return p0*x

	def poly1originC(x, p): # C stands for 'compact'
		return p[0]*x


	#----------------gaussian-------------------#

	def gaussian(x,amp,pos,sigma):
		return amp*np.exp(-(x-pos)**2/(2*(sigma**2)))

	def gaussianC(x,p):
		return p[0]*np.exp(-(x-p[1])**2/(2*(p[2]**2)))


	#------------------sinfit-------------------#

	def sinfit(x,a,b,c,d):
		return a*np.sin(b*x+c)+d

	def sinfitC(x,p):
		return p[0]*np.sin(p[1]*x+p[2])+p[3]


	#-----------------cosfit--------------------#

	def cosfit(x,a,b,c,d):
		return a*np.cos(b*x+c)+d

	def cosfitC(x,p):
		return p[0]*np.cos(p[1]*x+p[2])+p[3]


	#----------------exponential----------------#

	def exponential(x,a,b,c):
		return a*np.exp(b*x)+c

	def exponentialC(x,p):
		return p[0]*np.exp(p[1]*x)+p[2]


	#----------------lorentzian-----------------#

	def lorentzian(x,a,b,c):
		return a/((x-b)**2+(0.5*c)**2)

	def lorentzianC(x,p):
		return p[0]/((x-p[1])**2+(0.5*p[2])**2)


	#----------------cosecant-----------------#

	def cosecant(x,a,b,c,d):
		return a/(np.sin(b*x+c)) + d

	def cosecantC(x,p):
		return p[0]/(np.sin(p[1]*x+p[2])) + p[3]



	#----------------cosecant1-----------------#

	def cosecant1(x,a,b,c):
		return a/(np.sin(b*x)) + c

	def cosecant1C(x,p):
		return p[0]/(np.sin(p[1]*x)) + p[2]





	# def exponential, cosec function on one side, sin, cos, tan, cosh,




	## Polynomial Functions (0th to 9th order)


	#------------------0th----------------------#

	def poly0(x, p0):
		return p0

	def poly0C(x, p): # C stands for 'compact'
		return p[0]



	#------------------1st----------------------#

	def poly1(x, p0, p1):
		return p1*x + p0

	def poly1C(x, p): # C stands for 'compact'
		return p[1]*x + p[0]


	#------------------2nd----------------------#

	def poly2(x, p0, p1, p2):
		return p2*(x**2) + p1*x + p0

	def poly2C(x, p):
		return p[2]*(x**2) + p[1]*x + p[0]


	#------------------3rd----------------------#

	def poly3(x, p0, p1, p2, p3):
		return p3*(x**3) + p2*(x**2) + p1*x + p0

	def poly3C(x, p):
		return p[3]*(x**3) + p[2]*(x**2) + p[1]*x + p[0]


	#------------------4th----------------------#

	def poly4(x, p0, p1, p2, p3, p4):
		return p4*(x**4) + p3*(x**3) + p2*(x**2) + p1*x + p0

	def poly4C(x, p):
		return p[4]*(x**4) + p[3]*(x**3) + p[2]*(x**2) + p[1]*x + p[0]


	#------------------5th----------------------#

	def poly5(x, p0, p1, p2, p3, p4, p5):
		return p5*(x**5) + p4*(x**4) + p3*(x**3) + p2*(x**2) + p1*x + p0

	def poly5C(x, p):
		return p[5]*(x**5) + p[4]*(x**4) + p[3]*(x**3) + p[2]*(x**2) + p[1]*x + p[0]


	#------------------6th----------------------#

	def poly6(x, p0, p1, p2, p3, p4, p5, p6):
		return p6*(x**6) + p5*(x**5) + p4*(x**4) + p3*(x**3) + p2*(x**2) + p1*x + p0

	def poly6C(x, p):
		return p[6]*(x**6) + p[5]*(x**5) + p[4]*(x**4) + p[3]*(x**3) + p[2]*(x**2) + p[1]*x + p[0]


	#------------------7th----------------------#

	def poly7(x, p0, p1, p2, p3, p4, p5, p6, p7):
		return p7*(x**7) + p6*(x**6) + p5*(x**5) + p4*(x**4) + p3*(x**3) + p2*(x**2) + p1*x + p0

	def poly7C(x, p):
		return p[7]*(x**7) + p[6]*(x**6) + p[5]*(x**5) + p[4]*(x**4) + p[3]*(x**3) + p[2]*(x**2) + p[1]*x + p[0]


	#------------------8th----------------------#

	def poly8(x, p0, p1, p2, p3, p4, p5, p6, p7, p8):
		return p8*(x**8) + p7*(x**7) + p6*(x**6) + p5*(x**5) + p4*(x**4) + p3*(x**3) + p2*(x**2) + p1*x + p0

	def poly8C(x, p):
		return p[8]*(x**8) + p[7]*(x**7) + p[6]*(x**6) + p[5]*(x**5) + p[4]*(x**4) + p[3]*(x**3) + p[2]*(x**2) + p[1]*x + p[0]


	#------------------9th----------------------#

	def poly9(x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9):
		return p9*(x**9) + p8*(x**8) + p7*(x**7) + p6*(x**6) + p5*(x**5) + p4*(x**4) + p3*(x**3) + p2*(x**2) + p1*x + p0

	def poly9C(x, p):
		return p[9]*(x**9) + p[8]*(x**8) + p[7]*(x**7) + p[6]*(x**6) + p[5]*(x**5) + p[4]*(x**4) + p[3]*(x**3) + p[2]*(x**2) + p[1]*x + p[0]



	#------------------cos_squared----------------------#

	def cos_squared(x,a,b,c):
		return a*cos(x*b+c)**2+d

	def cos_squaredC(x,p):
		return p[0]*cos(x*p[1]+[2])**2+p[3]




	# Fit the function


	if modelname == 'poly1origin':

		p, cov_p = curve_fit(poly1origin, x, y, p0=initial_guess, sigma=errors, absolute_sigma=absolute_sigma, check_finite=True)

		fitted_function = poly1originC(x,p)


	if modelname == 'gaussian':

		p, cov_p = curve_fit(gaussian, x, y, p0=initial_guess, sigma=errors, absolute_sigma=absolute_sigma, check_finite=True)

		fitted_function = gaussianC(x,p)


	if modelname == 'sinfit':

		p, cov_p = curve_fit(sinfit, x, y, p0=initial_guess, sigma=errors, absolute_sigma=absolute_sigma, check_finite=True)

		fitted_function = sinfitC(x,p)


	if modelname == 'cosfit':

		p, cov_p = curve_fit(cosfit, x, y, p0=initial_guess, sigma=errors, absolute_sigma=absolute_sigma, check_finite=True)

		fitted_function = cosfitC(x,p)


	if modelname == 'exponential':

		p, cov_p = curve_fit(exponential, x, y, p0=initial_guess, sigma=errors, absolute_sigma=absolute_sigma, check_finite=True)

		fitted_function = exponentialC(x,p)


	if modelname == 'lorentzian':

		p, cov_p = curve_fit(lorentzian, x, y, p0=initial_guess, sigma=errors, absolute_sigma=absolute_sigma, check_finite=True)

		fitted_function = lorentzianC(x,p)


	if modelname == 'cosecant':

		p, cov_p = curve_fit(cosecant, x, y, p0=initial_guess, sigma=errors, absolute_sigma=absolute_sigma, check_finite=True)

		fitted_function = cosecantC(x,p)


	if modelname == 'cosecant1':

		p, cov_p = curve_fit(cosecant1, x, y, p0=initial_guess, sigma=errors, absolute_sigma=absolute_sigma, check_finite=True)

		fitted_function = cosecant1C(x,p)


	if modelname == 'poly0':

		p, cov_p = curve_fit(poly0, x, y, p0=initial_guess, sigma=errors, absolute_sigma=absolute_sigma, check_finite=True)

		fitted_function = poly0C(x,p)


	if modelname == 'poly1':

		p, cov_p = curve_fit(poly1, x, y, sigma=errors, absolute_sigma=absolute_sigma, check_finite=True)

		fitted_function = poly1C(x,p)

	if modelname == 'poly2':

		p, cov_p = curve_fit(poly2, x, y, p0=initial_guess, sigma=errors, absolute_sigma=absolute_sigma, check_finite=True)

		fitted_function = poly2C(x,p)

	if modelname == 'poly3':

		p, cov_p = curve_fit(poly3, x, y, p0=initial_guess, sigma=errors, absolute_sigma=absolute_sigma, check_finite=True)

		fitted_function = poly3C(x,p)

	if modelname == 'poly4':

		p, cov_p = curve_fit(poly4, x, y, p0=initial_guess, sigma=errors, absolute_sigma=absolute_sigma, check_finite=True)

		fitted_function = poly4C(x,p)

	if modelname == 'poly5':

		p, cov_p = curve_fit(poly5, x, y, p0=initial_guess, sigma=errors, absolute_sigma=absolute_sigma, check_finite=True)

		fitted_function = poly5C(x,p)

	if modelname == 'poly6':

		p, cov_p = curve_fit(poly6, x, y, p0=initial_guess, sigma=errors, absolute_sigma=absolute_sigma, check_finite=True)

		fitted_function = poly6C(x,p)

	if modelname == 'poly7':

		p, cov_p = curve_fit(poly7, x, y, p0=initial_guess, sigma=errors, absolute_sigma=absolute_sigma, check_finite=True)

		fitted_function = poly7C(x,p)

	if modelname == 'poly8':

		p, cov_p = curve_fit(poly8, x, y, p0=initial_guess, sigma=errors, absolute_sigma=absolute_sigma, check_finite=True)

		fitted_function = poly8C(x,p)

	if modelname == 'poly9':

		p, cov_p = curve_fit(poly9, x, y, p0=initial_guess, sigma=errors, absolute_sigma=absolute_sigma, check_finite=True)

		fitted_function = poly9C(x,p)


	if modelname == 'cos_squared':

		p, cov_p = curve_fit(cos_squared, x, y, p0=initial_guess, sigma=errors, absolute_sigma=absolute_sigma, check_finite=True)

		fitted_function = cos_squaredC(x,p)




	p_err = np.sqrt(np.diag(cov_p))


	chi_sq = red_chi2(y,fitted_function,p,errors=errors)


	return p, p_err, fitted_function, chi_sq
