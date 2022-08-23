

def boostrap_ci(pred, gs, ci=0.95, metric = "pearsonr"):
	""" Boostrapping to get confidence interval for prediction performance
	
	Params
	------
	pred_all: Numpy array
	ci: confidence interval
	
	Yields
	------
	cor_mean: float 
		middle bound
	lb: float
		lower bound
	ub: float
	upper bound
	
	"""
	import random
	import numpy as np
	import numpy.ma as ma

	def eva(x, y, metric = 'pearsonr'):
		'''
		metric: 'pearsonr', 'rmse'
		'''
		if metric =='pearsonr':
			return ma.corrcoef(ma.masked_invalid(x), ma.masked_invalid(y))[0,1]
		elif metric == 'rmse':
			return np.sqrt(np.mean((ma.masked_invalid(x)-ma.masked_invalid(y))**2))
		else:
			print("No metric "+metric+"!!!")
			
	# set random seed
	random.seed(42)

	pred_all = np.concatenate([[pred], [gs]], axis =0).T
	
	# calculate overall correlation
	cor_mean = eva(pred_all[:,0], pred_all[:,1], metric)
	#print("Overall prediction/gold standard correlation is %.4f" % cor_mean)
	# start boostrapping ...
	cor_all = [] 
	for i in range(100):
		pred_new = random.choices(pred_all, k = len(pred_all))
		pred_new = np.array(pred_new)
		cor = eva(pred_new[:,0], pred_new[:,1], metric)
		cor_all.append(cor)
	cor_all = sorted(cor_all)
	
	lb = cor_all[round(100*(0.5-ci*0.5))]
	ub = cor_all[round(100*(0.5+ci*0.5))]
	#print("%d%s Confidence interval is: (%.4f, %.4f)" % (int(ci*100), '%', lb, ub))
	
	return cor_mean, lb, ub
