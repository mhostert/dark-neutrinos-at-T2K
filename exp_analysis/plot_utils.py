############
# I created this little file, but feel free to merge with exp_analysis_class. 
# I just thought it was too busy there.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rc, rcParams
from matplotlib.pyplot import *
from matplotlib.backends.backend_pdf import PdfPages
import os

def kde_variable_plot(var1, var2, range, bins, m4mz, exp_analysis_obj, smoothing_pars=[0.005, 0.05], sel_criterion='no_cuts', selection='True', axis=False):
	# there is a faster way to do this by applying mask directly to the numpy array
	# -- but no guarantee indices match no_scan list 
	kde_df = exp_analysis_obj.df_base
	this_weight = exp_analysis_obj.kde_on_a_point(m4mz, smoothing_pars)
	# future implementation of something here?
	selection_weights = exp_analysis_obj.df_base.eval(selection)
	this_weight *= selection_weights

	this_weight = this_weight[kde_df[sel_criterion]]
	kde_df = kde_df[kde_df[sel_criterion]]


	
	kde_prediction, bin_edges = np.histogram(kde_df[var1, var2],
	         range=range,
	         bins=bins,
	         weights=this_weight,
	        )

	kde_errors2 = np.histogram(kde_df[var1, var2],
	             range=range,
	             bins=bins,
	             weights=this_weight**2,
	            )[0]

	# plotting
	if not axis:
		fsize=11
		fig = plt.figure()
		axes_form  = [0.14,0.15,0.82,0.76]
		ax = fig.add_axes(axes_form)
	else:
		ax = axis
	ax.plot(bin_edges,
	         np.append(kde_prediction, [0]),
	         ds='steps-post',
	         label=f'kde prediction: {kde_prediction.sum():.2g} '\
	            f'$\pm$ {100*np.sqrt(kde_errors2.sum())/kde_prediction.sum():.2g}%')

	kde_errors = np.sqrt(kde_errors2)
	for edge_left, edge_right, pred, err in zip(bin_edges[:-1], bin_edges[1:], kde_prediction, kde_errors):
		ax.add_patch(
			patches.Rectangle(
			(edge_left, pred-err),
			edge_right-edge_left,
			2 * err,
			hatch="\\\\\\\\\\",
			fill=False,
			linewidth=0,
			alpha=0.4,
			)
		)

	ax.legend(frameon=False, loc='best')
	ax.set_title(f'selection = {selection}')
	ax.set_xlabel(f'{var1} - {var2}')    
	ax.set_ylabel(f'Number of entries')

	return ax



def kde_to_noscan_comparison(var1, var2, range, bins, m4mz, exp_analysis_obj, smoothing_pars=[0.005, 0.05], sel_criterion='no_cuts', selection='True', axis=False):
	assert m4mz in list(exp_analysis_obj.dfs.keys())

	no_scan = exp_analysis_obj.dfs[m4mz]
	# there is a faster way to do this by applying mask directly to the numpy array 
	# -- but no guarantee indices match KDE list 
	no_scan = no_scan[no_scan[sel_criterion]]
	# mystery procedure
	selection_weights = no_scan.eval(selection)
	actual_weight = no_scan['actual_weight', '']
	this_weight = selection_weights * actual_weight
	
	no_scan_events = no_scan[var1, var2]

	no_scan_pred, bin_edges = np.histogram(no_scan_events,
	range=range,
	bins=bins,
	weights=this_weight,
	)
	no_scan_pred_err = np.histogram(no_scan_events,
	range=range,
	bins=bins,
	weights=this_weight**2,
	)[0]
	# plotting
	if not axis:
		fsize=11
		fig = plt.figure()
		axes_form  = [0.14,0.10,0.82,0.76]
		ax = fig.add_axes(axes_form)
	else:
		ax = axis
	# first KDE histogram with error bars
	kde_variable_plot(var1, var2, range, bins, m4mz, exp_analysis_obj, smoothing_pars=smoothing_pars, sel_criterion=sel_criterion, selection=selection, axis=ax)
	# now generated prediction
	ax.errorbar((bin_edges[1:]+bin_edges[:-1])/2, no_scan_pred, 
	yerr=np.sqrt(no_scan_pred_err),
	fmt='k.',
	label=f'no scanning: {no_scan_pred.sum():.2g} '\
	f'$\pm$ {100*np.sqrt(no_scan_pred_err.sum())/no_scan_pred.sum():.2g}%')

	ax.set_ylim(bottom=0,top=ax.get_ylim()[1])
	ax.set_xlim(left=0)
	ax.legend(frameon=False, loc='best')
	ax.set_title(f'selection = {selection} @ $m_4={m4mz[0]}$ GeV, $m_{{Z^\prime}}={m4mz[1]}$ GeV\n '\
	f'{exp_analysis_obj.hierarchy} {exp_analysis_obj.D_or_M[:3]}, {sel_criterion}\n '\
	f'smoothing pars = {smoothing_pars[0]} GeV, {smoothing_pars[1]} GeV')



def batch_comparison_plot(axes, exp_analyses, m4mz,var1,var2,smooth=(0.01,0.01),var_range=(0,1), bins=10, selection=True):
	
	kde_to_noscan_comparison(var1=var1, var2=var2, 
	                         range=var_range, bins=bins, 
	                         m4mz=m4mz, 
	                         exp_analysis_obj=exp_analyses[0],
	                         smoothing_pars=smooth, axis=axes[0], selection=selection)

	kde_to_noscan_comparison(var1=var1, var2=var2, 
	                         range=var_range, bins=bins, 
	                         m4mz=m4mz, 
	                         exp_analysis_obj=exp_analyses[1],
	                         smoothing_pars=smooth, axis=axes[1], selection=selection)


#######################################################
# streamlining the 4 panel plots -- currently only one hierarchy at a time
def batch_comparison_cutlevels(pdffilepath, exp_analyses, m4mz, smooth=(0.01,0.01), sel_criterion=False, 
								selection=True, variables=False, var_range=False, bins=False):

	if not os.path.isdir(os.path.basename(pdffilepath)):
		os.makedirs(os.path.basename(pdffilepath))

	# create pdf page...
	pdf = PdfPages(pdffilepath)

	if not variables:
		variables = [ ('ee_energy', ''),
					('ee_theta', ''),
					('ee_mass', ''),
					('ee_energy_asymetry', ''),
					('em_beam_theta', ''),
					('ep_beam_theta', ''),
					('experimental_t', '')]

	if not var_range:
		var_range = [(0,1.0),
					(0,np.pi/2),
					(0,m4mz[0]),
					(-1,1.0),
					(0,np.pi/2),
					(0,np.pi/2),
					(0,0.06)]
	if not bins:				
		bins = [10,
				10,
				10,
				10,
				10,
				10,
				10]
	elif np.size(bins)==1:
		bins = np.ones(np.size(variables))*bins
	
	if not sel_criterion:				
		sel_criterion = np.full(4,'no_cuts')
	elif np.size(sel_criterion)==1:
		sel_criterion = np.full(4,sel_criterion)

	###############
	# all variables
	for i in range(np.shape(variables)[0]):
		################
		# all four panels
		fig,  axes = plt.subplots(nrows=1, ncols=4,figsize = (18,4))
		for k in range(4):
			kde_to_noscan_comparison(var1=variables[i][0], var2=variables[i][1], 
									range=var_range[i], bins=bins[i], 
									m4mz=m4mz, sel_criterion=sel_criterion[k],
									exp_analysis_obj=exp_analyses[k],
									smoothing_pars=smooth, axis=axes[k], selection=selection)
		plt.tight_layout(); pdf.savefig(fig)
	plt.tight_layout()
	pdf.close()

#######################################################
# dirty function to plot everything...
def batch_comparison_light_heavy(pdffilepath, exp_analyses, m4mzheavy, m4mzlight, smooth=(0.01,0.01), selection=True):
	
	if not os.path.isdir(os.path.basename(pdffilepath)):
		os.makedirs(os.path.basename(pdffilepath))

	# create pdf page...
	pdf = PdfPages(pdffilepath)

	######################
	bins = 10
	var1='ee_energy'
	var2=''
	varmin=0; varmax=1.0
	fig,  axes = plt.subplots(nrows=1, ncols=4,figsize = (18,4))

	# Heavy
	exp_analyses_h=exp_analyses[:2]
	batch_comparison_plot(axes[:2],exp_analyses_h, m4mzheavy,var1,var2,smooth=smooth,var_range=(varmin,varmax), bins=bins)

	# light
	exp_analyses_l=exp_analyses[2:]
	batch_comparison_plot(axes[2:],exp_analyses_l, m4mzlight,var1,var2,smooth=smooth,var_range=(varmin,varmax), bins=bins)

	plt.tight_layout(); pdf.savefig(fig)

	######################
	bins = 10
	var1='ee_theta'
	var2=''
	varmin=0; varmax=np.pi/2
	fig,  axes = plt.subplots(nrows=1, ncols=4,figsize = (18,4))

	# Heavy
	exp_analyses_h=exp_analyses[:2]
	batch_comparison_plot(axes[:2],exp_analyses_h, m4mzheavy,var1,var2,smooth=smooth,var_range=(varmin,varmax), bins=bins)

	# light
	exp_analyses_l=exp_analyses[2:]
	batch_comparison_plot(axes[2:],exp_analyses_l, m4mzlight,var1,var2,smooth=smooth,var_range=(varmin,varmax), bins=bins)

	plt.tight_layout(); pdf.savefig(fig)

	######################
	bins = 5
	var1='ee_mass'
	var2=''
	fig,  axes = plt.subplots(nrows=1, ncols=4,figsize = (18,4))

	# Heavy
	varmin=0; varmax=m4mzheavy[0]
	exp_analyses_h=exp_analyses[:2]
	batch_comparison_plot(axes[:2],exp_analyses_h, m4mzheavy,var1,var2,smooth=smooth,var_range=(varmin,varmax), bins=bins)

	# light
	varmin=0; varmax=m4mzlight[0]
	exp_analyses_l=exp_analyses[2:]
	batch_comparison_plot(axes[2:],exp_analyses_l, m4mzlight,var1,var2,smooth=smooth,var_range=(varmin,varmax), bins=bins)

	plt.tight_layout(); pdf.savefig(fig)

	######################
	bins = 20
	var1='ee_energy_asymetry'
	var2=''
	varmin=-1; varmax=1
	fig,  axes = plt.subplots(nrows=1, ncols=4,figsize = (18,4))

	# Heavy
	exp_analyses_h=exp_analyses[:2]
	batch_comparison_plot(axes[:2],exp_analyses_h, m4mzheavy,var1,var2,smooth=smooth,var_range=(varmin,varmax), bins=bins)

	# light
	exp_analyses_l=exp_analyses[2:]
	batch_comparison_plot(axes[2:],exp_analyses_l, m4mzlight,var1,var2,smooth=smooth,var_range=(varmin,varmax), bins=bins)

	plt.tight_layout(); pdf.savefig(fig)

	######################
	bins = 20
	var1='em_beam_theta'
	var2=''
	varmin=0; varmax=np.pi
	fig,  axes = plt.subplots(nrows=1, ncols=4,figsize = (18,4))

	# Heavy
	exp_analyses_h=exp_analyses[:2]
	batch_comparison_plot(axes[:2],exp_analyses_h, m4mzheavy,var1,var2,smooth=smooth,var_range=(varmin,varmax), bins=bins)

	# light
	exp_analyses_l=exp_analyses[2:]
	batch_comparison_plot(axes[2:],exp_analyses_l, m4mzlight,var1,var2,smooth=smooth,var_range=(varmin,varmax), bins=bins)

	plt.tight_layout(); pdf.savefig(fig)


	######################
	var1='ep_beam_theta'
	fig,  axes = plt.subplots(nrows=1, ncols=4,figsize = (18,4))

	# Heavy
	exp_analyses_h=exp_analyses[:2]
	batch_comparison_plot(axes[:2],exp_analyses_h, m4mzheavy,var1,var2,smooth=smooth,var_range=(varmin,varmax), bins=bins)

	# light
	exp_analyses_l=exp_analyses[2:]
	batch_comparison_plot(axes[2:],exp_analyses_l, m4mzlight,var1,var2,smooth=smooth,var_range=(varmin,varmax), bins=bins)

	plt.tight_layout(); pdf.savefig(fig)

	######################
	bins = 20
	var1='experimental_t'
	var2=''
	varmin=0; varmax=0.06
	fig,  axes = plt.subplots(nrows=1, ncols=4,figsize = (18,4))

	# Heavy
	exp_analyses_h=exp_analyses[:2]
	batch_comparison_plot(axes[:2],exp_analyses_h, m4mzheavy,var1,var2,smooth=smooth,var_range=(varmin,varmax), bins=bins)

	# light
	exp_analyses_l=exp_analyses[2:]
	batch_comparison_plot(axes[2:],exp_analyses_l, m4mzlight,var1,var2,smooth=smooth,var_range=(varmin,varmax), bins=bins)

	plt.tight_layout(); pdf.savefig(fig)

	pdf.close()

