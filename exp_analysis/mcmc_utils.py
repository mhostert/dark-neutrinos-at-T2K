import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import emcee
import corner
from getdist import plots, MCSamples

from dark_nus_utils import load_datasets
from parameters_dict import physics_parameters

labels_fancy = {
    'm4': r'$m_4$ [GeV]',
    'mz': r'$m_Z$ [GeV]', 
    'Vmu4_alpha_epsilon2': r'$V_{\mu 4}\alpha \epsilon^2$',
    'Valpha4_alpha_epsilon2': r'$V_{\alpha 4}\alpha \epsilon^2$',

    'Vmu4': r'$V_{\mu 4}$',
    'epsilon': r'$\epsilon$',

    'log(p)': r'log(p)',
    'ctau': r'$c\tau$ [cm]',
    
    'mu': r'$\mu$',
    'sigma2': r'$\sigma^2$',
    'sigma/mu': r'$\sigma/\mu$',
}

loglabels_fancy = {}
for key, label in labels_fancy.items():
    loglabels_fancy['log10_' + key] = label.replace('$', '$\log_{10}', 1)
labels_fancy.update(loglabels_fancy)

class dark_nus_mcmc(object):
    def __init__(self, hierarchy, D_or_M, title_addition, direct_load_objects=True):
        self.hierarchy = hierarchy
        self.D_or_M = D_or_M
        self.exp_analysis_objs = load_datasets(hierarchy,
                                                D_or_M,
                                                timeit=True,
                                                direct_load_objects=direct_load_objects)
        self.__dict__.update(physics_parameters[hierarchy])

        self.title_addition = title_addition
        self.save_folder = f"../fig/5_mcmc/{self.hierarchy}_{self.D_or_M}_{self.title_addition.strip().replace(' ', '_')}/"
        if not(os.path.exists(self.save_folder) and os.path.isdir(self.save_folder)):
            os.makedirs(self.save_folder)

    def points_on_triangle(self, N_points, log=False):
        rvs = np.random.random((N_points, 2))
        if self.hierarchy == 'heavy':
            rvs = np.where(rvs[:, 0, None]<rvs[:, 1, None], rvs, rvs[:, ::-1])
        elif self.hierarchy == 'light':
            rvs = np.where(rvs[:, 0, None]>rvs[:, 1, None], rvs, rvs[:, ::-1])
        
        if not log:
            m4_limits = self.m4_limits
            mz_limits = self.mz_limits
        else:
            m4_limits = [np.log10(lim) for lim in self.m4_limits]
            mz_limits = [np.log10(lim) for lim in self.mz_limits]
        return np.array((m4_limits[0], mz_limits[0])) + rvs*(m4_limits[1]-m4_limits[0], mz_limits[1]-mz_limits[0])

    def set_posterior(self, posterior, ndim, labels):
        self.posterior = posterior
        self.ndim = ndim
        self.labels = labels
        self.title = f'{self.hierarchy} {self.D_or_M}, {self.title_addition}'

    def initialise_mcmc(self, nwalkers, pool, blobs_dtype=None, set_backend=False, reset_backend=True):
        self.reset_backend = reset_backend
        if self.reset_backend:
            for i in range(nwalkers):
                print(self.p0[i], self.posterior(self.p0[i]))

        if set_backend:
            self.posterior_filename = f"./posteriors/{self.hierarchy}_{self.D_or_M}_{self.title_addition.strip().replace(' ', '_')}.h5"
            self.backend = emcee.backends.HDFBackend(self.posterior_filename)
            if self.reset_backend:
                self.backend.reset(nwalkers, self.ndim)
        else:
            self.backend = None
        self.blobs_dtype = blobs_dtype
        self.sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.posterior, pool=pool, blobs_dtype=blobs_dtype, backend=self.backend)

    def run_mcmc(self, n_samples):
        if self.reset_backend:
            self.current_state = self.sampler.run_mcmc(self.p0, n_samples, progress=True)
        else:
            self.current_state = self.sampler.run_mcmc(None, n_samples, progress=True)

    def compute_autocorrelation_time(self, discard=0, thin=1, store=False):
        chain = self.sampler.get_chain(discard=discard, thin=thin)
        print(chain.shape)
        len_chain = len(chain)
        auto_corr_times = []
        indices = np.geomspace(1, len_chain, 15).astype(int)
        for i in indices:
            auto_corr_times.append(emcee.autocorr.integrated_time(chain[:i],
                                        c=5,
                                        tol=50,
                                        quiet=True,
                                        ))
        auto_corr_times = np.array(auto_corr_times).T
        for auto_corr_time, label in zip(auto_corr_times, self.labels):
            plt.plot(indices, auto_corr_time, label=labels_fancy[label])
        plt.plot(indices, indices/100, "--k", label='N/50')
        plt.xlabel("number of steps")
        plt.ylabel(r"mean correlation time")
        plt.legend(frameon=False)
        plt.title(self.title)
        if store:
            plt.savefig(self.save_folder + 'autocorrelation.pdf', bbox_inches='tight')

    def get_samples(self, discard=0, thin=1):
        tau = self.sampler.get_autocorr_time(quiet=True)
        discard_int = int(tau.max()*discard)
        thin_int = 1
        if thin != 1:
            thin_int = int(tau.min()*thin)

        chains = self.sampler.get_chain(discard=discard_int, thin=thin_int)
        column_list = self.labels.copy()

        log_probs = self.sampler.get_log_prob(discard=discard_int, thin=thin_int)
        column_list.append('log(p)')

        self.raw_chains = np.concatenate([chains, log_probs[..., np.newaxis]], axis=-1)
        if self.blobs_dtype is not None:
            blobs = self.sampler.get_blobs(discard=discard_int, thin=thin_int)
            for dtype in self.blobs_dtype:
                self.raw_chains = np.concatenate([self.raw_chains, blobs[dtype[0]][..., np.newaxis]], axis=-1)
                column_list.append(dtype[0])
        samples = self.raw_chains.reshape(-1, self.raw_chains.shape[-1])
        self.samples = pd.DataFrame(samples, columns=column_list)
        self.samples['sigma/mu'] = np.sqrt(self.samples['sigma2'])/self.samples['mu']
        self.samples['log10_mu'] = np.log10(self.samples['mu'])
        self.samples['log10_sigma/mu'] = np.log10(self.samples['sigma/mu'])
        if self.hierarchy == 'heavy':
            self.samples['log10_ctau'] = np.log10(self.samples['ctau'])

        if 'm4' in self.labels:
            self.samples['log10_m4'] = np.log10(self.samples['m4'])
        elif 'log10_m4' in self.labels:
            self.samples['m4'] = 10**(self.samples['log10_m4'])
        if 'mz' in self.labels:
            self.samples['log10_mz'] = np.log10(self.samples['mz'])
        elif 'log10_mz' in self.labels:
            self.samples['mz'] = 10**(self.samples['log10_mz'])

    def plot_chains(self, store=False):
        fig, axes = plt.subplots(self.ndim+1, figsize=((self.ndim+1)*3, 7), sharex=True)
        plt.suptitle(self.title)
        for i in range(self.ndim):
            ax = axes[i]
            ax.plot(self.raw_chains[..., i], alpha=0.3)
            ax.set_ylabel(labels_fancy[self.labels[i]])
        axes[-1].plot(self.raw_chains[..., -1], alpha=0.3)
        axes[-1].set_ylabel("log(p)")
        axes[-1].set_xlabel("step number")
        if store:
            plt.savefig(self.save_folder + 'raw_chains.pdf', bbox_inches='tight')
        return fig, axes

    def corner_plot(self, which_labels, levels=[0.13], savefile=None):
        samples = self.samples[which_labels].values
        corner_plot_labels = [labels_fancy[label] for label in which_labels]
        corner.corner(samples, 
                      levels=levels,
                      labels=corner_plot_labels)
        plt.suptitle(self.title)
        if savefile is not None:
            plt.savefig(self.save_folder + savefile + '.pdf', bbox_inches='tight')
            
    def corner_plot_raw(self, levels=[0.13], savefile=None):
        self.corner_plot(which_labels=self.labels + ['log(p)'],
                         levels=levels,
                         savefile=savefile)

    def corner_plot_with_colour(self, colour_variable, savefile=None):
        ranges = {'m4': self.m4_limits,
                  'mz': self.mz_limits,
                  'log10_m4': [np.log10(lim) for lim in self.m4_limits],
                  'log10_mz': [np.log10(lim) for lim in self.mz_limits],
                  'log10_Vmu4_alpha_epsilon2': [np.log10(self.lower_bound_Vmu4_alpha_epsilon2), 
                                                np.log10(self.upper_bound_Vmu4_alpha_epsilon2)],
                 }
        if self.hierarchy == 'heavy':
            ranges['log10_Valpha4_alpha_epsilon2'] = [np.log10(self.lower_bound_Vmu4_alpha_epsilon2), 
                                                      np.log10(self.upper_bound_Valpha4_alpha_epsilon2)]
        samples = MCSamples(samples=self.samples.values,
                    names=self.samples.columns,
                    labels=[labels_fancy[lab].replace('$', '') if lab in labels_fancy.keys() else lab for lab in self.samples.columns],
                    ranges=ranges,
                    settings={'boundary_correction_order':0, 'mult_bias_correction_order':1.})
        g = plots.get_subplot_plotter()
        g.triangle_plot(samples,
                        self.labels,
                        
                        plot_3d_with_param=colour_variable)
        plt.suptitle(self.title)
        if savefile is not None:
            plt.savefig(self.save_folder + savefile + '.pdf', bbox_inches='tight')

# class heavy_minimal_mcmc(dark_nus_mcmc):

#     def initialise_mcmc(self, nwalkers, pool, blobs_dtype=None, set_backend=False, reset_backend=True, log_ms=False):
#         self.log_ms = log_ms
#         while reset_backend:
#             m4_mz_0 = self.points_on_triangle(nwalkers, log=self.log_ms)
#             log10_Vmu4_alpha_epsilon2_0 = np.random.uniform(np.log10(self.exp_analysis_obj.lower_bound_Vmu4_alpha_epsilon2),
#                                                             np.log10(self.exp_analysis_obj.upper_bound_Vmu4_alpha_epsilon2),
#                                                             nwalkers)
#             p0 = np.column_stack([m4_mz_0, log10_Vmu4_alpha_epsilon2_0])
#             start_posterior = []
#             for i in range(nwalkers):
#                 start_posterior.append(self.posterior(p0[i]))
#             if np.isinf(start_posterior).sum() == 0:
#                 self.p0 = p0
#                 break

#         super().initialise_mcmc(nwalkers, pool, blobs_dtype, set_backend, reset_backend)

#     def corner_plot_physics(self, levels=[0.13], savefile=None):
#         super().corner_plot(which_labels=[f"{'log10_' if self.log_ms else ''}m4", f"{'log10_' if self.log_ms else ''}mz", 'log10_Vmu4_alpha_epsilon2', 'log10_ctau'], levels=levels, savefile=savefile)
        
class heavy_nonminimal_mcmc(dark_nus_mcmc):

    def initialise_mcmc(self, nwalkers, pool, blobs_dtype=None, set_backend=False, reset_backend=True, log_ms=False):
        self.log_ms = log_ms
        while reset_backend:
            print('new trial to set up start points')
            m4_mz_0 = self.points_on_triangle(nwalkers, log=self.log_ms)
            log10_Vmu4_alpha_epsilon2_0 = np.random.uniform(np.log10(self.lower_bound_Vmu4_alpha_epsilon2),
                                                            np.log10(self.upper_bound_Vmu4_alpha_epsilon2),
                                                            nwalkers)
            log10_Valpha4_alpha_epsilon2_0 = np.random.uniform(log10_Vmu4_alpha_epsilon2_0,
                                                     np.log10(self.upper_bound_Valpha4_alpha_epsilon2))
            p0 = np.column_stack([m4_mz_0, log10_Vmu4_alpha_epsilon2_0, log10_Valpha4_alpha_epsilon2_0])
            start_posterior = []
            for i in range(nwalkers):
                start_posterior.append(self.posterior(p0[i]))
            if np.isinf(start_posterior).sum() == 0:
                self.p0 = p0
                break
        
        super().initialise_mcmc(nwalkers, pool, blobs_dtype, set_backend, reset_backend)

    def corner_plot_physics(self, levels=[0.13], savefile=None):
        super().corner_plot(which_labels=[f"{'log10_' if self.log_ms else ''}m4", f"{'log10_' if self.log_ms else ''}mz", 'log10_Vmu4_alpha_epsilon2', 'log10_Valpha4_alpha_epsilon2', 'log10_ctau'], levels=levels, savefile=savefile)


class light_minimal_mcmc(dark_nus_mcmc):

    def initialise_mcmc(self, nwalkers, pool, blobs_dtype=None, set_backend=False, reset_backend=True, log_ms=False):
        self.log_ms = log_ms
        
        while reset_backend:
            print('new trial to set up start points')
            m4_mz_0 = self.points_on_triangle(nwalkers, log=self.log_ms)
            log10_Vmu4_alpha_epsilon2 = np.random.uniform(np.log10(self.lower_bound_Vmu4_alpha_epsilon2),
                                           np.log10(self.upper_bound_Vmu4_alpha_epsilon2),
                                           nwalkers)
            p0 = np.column_stack([m4_mz_0, log10_Vmu4_alpha_epsilon2])
            start_posterior = []
            for i in range(nwalkers):
                start_posterior.append(self.posterior(p0[i]))
            if np.isinf(start_posterior).sum() == 0:
                self.p0 = p0
                break

        super().initialise_mcmc(nwalkers, pool, blobs_dtype, set_backend, reset_backend)

    def corner_plot_physics(self, levels=[0.13], savefile=None):
        super().corner_plot(which_labels=[f"{'log10_' if self.log_ms else ''}m4", f"{'log10_' if self.log_ms else ''}mz", 'log10_Vmu4_alpha_epsilon2'], levels=levels, savefile=savefile)