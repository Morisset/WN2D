#%% 
# imports
import numpy as np
import matplotlib.pyplot as plt
import pyneb as pn
import pyCloudy as pc

from pyCloudy.utils.physics import abunds_Bressan_2012, get_abund_nicholls, ATOMIC_MASS, Z as Z_pc
from pyCloudy.utils.physics import depletion_dopita_2013, depletion_cloudy_17

#%% 
# configuration
pc.config.cloudy_exe = '/usr/local/Cloudy/c25.00/source/cloudy.exe'
dir_models = './models/'
#pc.print_make_file(dir_models)

#%%
# PyNeb stuff
pn.config.use_multiprocs()
O3 = pn.Atom('O', 3)
N2 = pn.Atom('N', 2)
O2 = pn.Atom('O', 2)
Cl3 = pn.Atom('Cl', 3)

#%% 
# functions

def deplete(ab_dic, depl_dic=None, log_depl = 0.0, ksi_d=None):
    
    if ksi_d is not None:
        Z, Z_dep, ksi_d_ori, Y = get_metallicity(ab_dic, depl_dic, log_depl, ksi_d=None)
        #print(ksi_d_ori)
    new_dic = {}
    for elem in ab_dic:
        new_dic[elem] = ab_dic[elem]
        if elem not in ('H' 'He'):
            if depl_dic is not None:
                if elem in depl_dic:
                    if ksi_d is not None:
                        if ksi_d <= ksi_d_ori: # lim ksi_d = 0 -> fdepl = 0; ksi_d = kdi_d_ori -> fdepl = (1-depl_dic)
                            fdepl = ksi_d/ksi_d_ori * (1-depl_dic[elem])
                        else: # lim ksi_d = 1 -> fdepl = 1; ksi_d = kdi_d_ori -> fdepl = (1-depl_dic)
                            fdepl = 1 - depl_dic[elem] * (1-ksi_d) / (1-ksi_d_ori)
                        new_dic[elem] +=  np.log10((1 - fdepl))
                    else:
                        new_dic[elem] += np.log10(depl_dic[elem])
            new_dic[elem] += log_depl 
    return new_dic

def get_metallicity(ab_dic, depl_dic=None, log_depl = 0.0, ksi_d=None):
    """
    Return the total metallicity, the metallicity in gaseous phase, the ksi_d from:
        ab_dic: dictionnary of abundances in log X/H
        depl_dic: dictionnary of depletion as factors
        log_depl: [dex] a factor applied to the depletion to adjust the Ksi_d value
    """
    
    M_metals = 0
    M_metals_dep = 0
    M_all = ATOMIC_MASS['H']
    M_all_dep = ATOMIC_MASS['H'] 
    M_Helio = 0
    dep_dic = deplete(ab_dic, depl_dic=depl_dic, log_depl = log_depl, ksi_d=ksi_d)
    for elem in dep_dic:
        if elem != 'H':
            to_add = 10**ab_dic[elem] * ATOMIC_MASS[elem]
            to_add_dep = 10**dep_dic[elem] * ATOMIC_MASS[elem]
            M_all += to_add
            M_all_dep += to_add_dep
            if elem != 'He':
                M_metals += to_add
                M_metals_dep += to_add_dep
            else:
                M_Helio += to_add
    return M_metals / M_all, M_metals_dep / M_all_dep, 1 - M_metals_dep/M_metals, M_Helio/M_all
        

#%% 
# get_abunds

def get_abunds(log_OH_tot=-3.16526, ksi_d=0):
    
    abund = get_abund_nicholls(log_OH_tot)
    depletion = depletion_cloudy_17
    for elem in depletion_dopita_2013:
        depletion[elem] = depletion_dopita_2013[elem]
    
    Z, Z_dep, ksi_d2, Y = get_metallicity(abund, depletion, log_depl=0, ksi_d=ksi_d)
    abund2 = deplete(abund, depletion, log_depl=0, ksi_d=ksi_d)
    
    return Z, Z_dep, ksi_d2, abund, abund2

def logU_OH(log_OH):
    return -6.5 - log_OH


def Kewley(K_x, K_x_max=0.47):
    """
    Kewley relation
    x = log([NII] 6584/ Ha)
    return y in log([OIII]5007 / Hb
    """
    y = 0.61 / (K_x - K_x_max) + 1.19
    return np.where(K_x < K_x_max, y, -np.inf)

def Kauffman(K_x, K_x_max=0.03):
    """
    Kauffman relation
    x = log([NII] 6584/ Ha)
    return y in log([OIII]5007 / Hb
    """
    y = 0.61 / (K_x - K_x_max) + 1.3
    return np.where(K_x < K_x_max, y, -np.inf)

def Stasinska(x):
    """
    Stasinska 06 relation
    """

    y = (-30.787 + 1.1358*x + 0.27297*x**2) * np.tanh(5.7409*x) - 31.093
    return y

emis_tab = ['H  1 4861.32A',
            'Ca B 4861.32A',
            'H  1 6562.80A',
            'Ca B 6562.80A',
            'N  2 5754.59A',
            'N 2R 5755.00A',
            'N 2T 5755.00A',
            'N  2 6548.05A',
            'N  2 6583.45A',
            'O  1 6300.30A',
            'O  2 3726.03A',
            'O  2 3728.81A',
            'O  2 7318.92A',
            'O  2 7319.99A',
            'O  2 7329.67A',
            'O  2 7330.73A',
            'Blnd 7323.00A',
            'Blnd 7332.00A',
            'O 2R 3726.00A',
            'O 2R 3729.00A',
            'O 2R 7323.00A',
            'O 2R 7332.00A',
            'BLND 3727.00A',
            'BLND 7325.00A',
            'O  3 4363.21A',
            'O 3C 4363.00A',
            'O 3R 4363.00A',
            'O  3 4958.91A',
            'O  3 5006.84A',
            'O  3 51.8004M',
            'O  3 88.3323M',
            'O  3 1660.81A',
            'O  3 1666.15A',
            'O  3 3059.28A',
            'O  3 3047.10A',
            'Ne 3 3868.76A',
            'Ne 3 3967.47A',
            'Ne 3 1814.56A',
            'Ne 3 4011.68A',
            'Ne 3 15.5509M',
            'Ne 3 36.0036M',
            'S  2 4068.60A',
            'S  2 4076.35A',
            'S  2 6730.82A',
            'S  2 6716.44A',
            'S  3 6312.06A',
            'S  3 9530.62A',
            'S  3 9068.62A',
            'S  3 18.7078M',
            'S  3 33.4704M',
            'Ar 3 7135.79A',
            'Ar 4 7170.70A',
            'Ar 4 7263.33A',
            'Ar 4 4711.26A',
            'Ar 4 4740.12A',
            'Cl 3 5517.71A',
            'Cl 3 5537.87A',
            'O  1 63.1679m',
            'O  1 145.495m',
            'C  2 157.636m'
]

#%% 
# make models

def make_model(name, log_OH, fr=0.3, NH=10, ff=1, ksi_d=0.35, age=1, 
               verbose=True, atm_file = 'BPASSv2.2.1_bin-imf_chab300.idx'):


    c = pc.CST.CLIGHT
    alpha_B = 2.6e-13

    logU = logU_OH(log_OH)

    Z_sol = 0.01425
    w = (1 + fr**3.)**(1./3) - fr
    U = 10**logU
    Q0 = 4. * np.pi * c**3 * U**3 / (3. * NH * ff**2 * alpha_B**2 * w**3)
    R_str = (3. * Q0 / (4 * np.pi * NH**2 * alpha_B * ff))**(1./3)
    R_in = fr * R_str

    Z, Z_dep, ksi_d2, abund_ism, abund = get_abunds(log_OH, ksi_d)
    
    dtg = ksi_d2 / 0.36 * Z / Z_sol # log10(0.01524) = -1.817015

    metallicity = min(max(np.log10(Z), -4.99), -1.41)
    if verbose:
        print(f'Z = {Z:.3f}, Z_dep = {Z_dep:.3f}, ksi_d = {ksi_d2:.3f}, dtg = {dtg:.3f}')
        print(f'logU = {logU:.3f}, Q0 = {Q0:.2e}, log R_in = {np.log10(R_in):.2f} cm')
           
    full_name = dir_models + name
    Min = pc.CloudyInput(full_name)


    Min.set_radius(r_in = np.log10(R_in))
    Min.set_cste_density(dens = np.log10(NH))
    Min.set_abund(ab_dict = abund)
    Min.set_grains('ism linear {0}'.format(dtg)) 

    Min.set_star(SED = f'table stars "{atm_file}"', SED_params = (age*1e6, metallicity), 
                    lumi_unit= 'q(H)', lumi_value = np.log10(Q0))
 
    Min.set_other(('COSMIC RAY BACKGROUND',
                    'CMB'))
    Min.set_iterate(1)
    Min.set_sphere(True)
    Min.set_stop(('temperature 20', 'pfrac 0.02'))
    Min.set_emis_tab(emis_tab)
    Min.print_input()

def make_grid(name='M5_2My_', log_OH_min=-5.0, log_OH_max=-3.0, n_points=17, 
              age=2, run=True, verbose=True, n_proc=10, fr=0.3, NH=10, ff=1, ksi_d=0.35, atm_file='BPASSv2.2.1_bin-imf_chab300.idx'):

    for log_OH in np.linspace(log_OH_min, log_OH_max, n_points):
        make_model(name=f'{name}{log_OH:.1f}', log_OH=log_OH, age=age, fr=fr, NH=NH, ff=ff, ksi_d=ksi_d, verbose=verbose, atm_file=atm_file)
    if run:
        pc.run_cloudy(dir_ = dir_models, n_proc = n_proc, use_make = True, model_name = name)


#%% 
# read models

l_dict = {'Has': r'H$\alpha$',
        'Hbs': r'H$\beta$',
        'N2s': r'[NII]',
        'O2s': r'[OII]',
        'N2as': r'[NII] 5755',
        'O3s': r'[OIII]',
        'O3as': r'[OIII] 4363',
        'R23s': r'[OII] + [OIII]',
        'rO3s': r'[OIII] 4363/5007',
        'Ar3O3': r'[ArIII]/[OIII]',
        'S3O3': r'[SIII]/[OIII]',
        'S2s': r'[SII]',
        'S3s': r'[SIII]',
        'Ar3s': r'[ArIII]',
        'O1s': r'[OI]',
        'Cl3s': r'[ClIII]',
        'logUs': r'log(U)',
        'logQ0s': r'log(Q0)',
        'logRins': r'log(Rin)',
        'OHs': r'log(O/H)',
        '12OHs': r'12 + log(O/H)',
        'NOs': r'log(N/O)',
        'Top': r'T(O^{+})',
        'Topp': r'T(O^{++})',
        'N2S2Ha': r'log([NII]/[SII]) + 0.264 * log([NII]/H$\alpha$)',
        'pR2s': r'[OII]/H$\beta$',
        'pN2s': r'[NII]/H$\beta$',
        'pS2s': r'[SII]/H$\beta$',
        'pR3s': r'[OIII]/H$\beta$',
        'pOH_R': r'12+log O/H (Pil R)',
        'pOH_S': r'12+log O/H(Pil S)',
        }

class model_grid(object):

    def __init__(self, name, dims=100, cut=None, limit=0.05, dir_models=dir_models, remove_NIP=True):
        self.name = name
        self.Ms = pc.load_models(model_name=dir_models + self.name)
        if remove_NIP:
                self.Ms = [M for M in self.Ms if M.model_name_s.split('_')[-1] != 'NIP']
        self.Ms = sorted(self.Ms, key=lambda M: M.abund['O'])
        if cut is not None and cut > 0 and cut < 1:
            for M in self.Ms:
                M.Hbeta_cut = M.Hbeta * cut
        self.set_integ()
        self.dims = dims
        self.set_2D(dims=[dims, dims, 1])
        self.set_mask(limit=limit)

    def set_integ(self):

        self.I1 = {}
        self.I1['Has'] = np.asarray([M.get_emis_vol('H__1_656280A') for M in self.Ms])
        self.I1['Hbs'] = np.asarray([M.get_emis_vol('H__1_486132A') for M in self.Ms])
        self.I1['N2s'] = np.asarray([M.get_emis_vol('N__2_658345A') for M in self.Ms])
        self.I1['N2as'] = np.asarray([M.get_emis_vol('N__2_575459A') + M.get_emis_vol('N_2R_575500A') for M in self.Ms])
        self.I1['O2s'] = np.asarray([M.get_emis_vol('O__2_372603A') + M.get_emis_vol('O__2_372881A') for M in self.Ms])
        self.I1['O3s'] = np.asarray([M.get_emis_vol('O__3_500684A') for M in self.Ms])
        self.I1['O3as'] = np.asarray([M.get_emis_vol('O__3_436321A') + M.get_emis_vol('O_3R_436300A') for M in self.Ms])
        self.I1['R23s'] = np.asarray([1.3*M.get_emis_vol('O__3_500684A') + M.get_emis_vol('O__2_372603A') + M.get_emis_vol('O__2_372881A') for M in self.Ms])
        self.I1['rO3s'] = np.asarray([(M.get_emis_vol('O__3_436321A') + M.get_emis_vol('O_3R_436300A')) / M.get_emis_vol('O__3_500684A') for M in self.Ms])
        self.I1['S2s'] = np.asarray([M.get_emis_vol('S__2_671644A') + M.get_emis_vol('S__2_673082A') for M in self.Ms])
        self.I1['S3s'] = np.asarray([M.get_emis_vol('S__3_906862A') for M in self.Ms])
        self.I1['Ar3s'] = np.asarray([M.get_emis_vol('AR_3_713579A') for M in self.Ms])
        self.I1['pR2s'] = self.I1['O2s'] / self.I1['Hbs']
        self.I1['pN2s'] = 1.33 * self.I1['N2s'] / self.I1['Hbs']
        self.I1['pS2s'] = self.I1['S2s'] / self.I1['Hbs']
        self.I1['pR3s'] = 1.33 * self.I1['O3s'] / self.I1['Hbs']

        self.I1['pOH_RU'] = (8.589 + 
                             0.022 * np.log10(self.I1['pR3s']/self.I1['pR2s']) + 
                             0.399 * np.log10(self.I1['pN2s']) + 
                             np.log10(self.I1['pR2s']) * 
                                (-0.137 + 0.164 * np.log10(self.I1['pR3s']/self.I1['pR2s']) + 0.589 * np.log10(self.I1['pN2s']))
                             )
        self.I1['pOH_RL'] = (7.932 +
                             0.944 * np.log10(self.I1['pR3s']/self.I1['pR2s']) + 
                             0.695 * np.log10(self.I1['pN2s']) + 
                             np.log10(self.I1['pR2s']) * 
                                (0.970 - 0.291 * np.log10(self.I1['pR3s']/self.I1['pR2s']) - 0.019 * np.log10(self.I1['pN2s']))
                            )
        self.I1['pOH_R'] = np.where(np.log10(self.I1['pR2s']) > -0.6, self.I1['pOH_RU'], self.I1['pOH_RL'])
        self.I1['pOH_SU'] = (8.424 +
                             0.030 * np.log10(self.I1['pR3s']/self.I1['pS2s']) + 
                             0.751 * np.log10(self.I1['pN2s']) + 
                             np.log10(self.I1['pS2s']) * 
                                (-0.349 + 0.182 * np.log10(self.I1['pR3s']/self.I1['pS2s']) + 0.508 * np.log10(self.I1['pN2s']))
                            )
        self.I1['pOH_SL'] = (8.072 +
                             0.789 * np.log10(self.I1['pR3s']/self.I1['pS2s']) + 
                             0.726 * np.log10(self.I1['pN2s']) + 
                             np.log10(self.I1['pR2s']) * 
                                (1.069 - 0.170 * np.log10(self.I1['pR3s']/self.I1['pS2s']) - 0.022 * np.log10(self.I1['pN2s']))
                            )
        self.I1['pOH_S'] = np.where(np.log10(self.I1['pN2s']) > -0.6, self.I1['pOH_SU'], self.I1['pOH_SL'])
        self.I1['O1s'] = np.asarray([M.get_emis_vol('O__1_630030A') for M in self.Ms])
        self.I1['Cl3s'] = np.asarray([(M.get_emis_vol('CL_3_551771A') + M.get_emis_vol('CL_3_553787A')) for M in self.Ms])
        self.I1['N2S2Ha'] = 10**(np.log10(self.I1['N2s'] / self.I1['S2s']) + 0.264 * np.log10(self.I1['N2s'] / self.I1['Has']))
        self.I1['OHs'] = np.asarray([M.abund['O'] for M in self.Ms])
        self.I1['12OHs'] = 12 + self.I1['OHs']
        self.I1['logQ0s'] = np.asarray([np.log10(M.Q0) for M in self.Ms])
        self.I1['logUs'] = np.asarray([M.log_U_mean for M in self.Ms])
        self.I1['logRins'] = np.asarray([np.log10(M.r_in) for M in self.Ms])
        self.I1['NOs'] = np.asarray([M.abund['N'] - M.abund['O'] for M in self.Ms])
        self.I1['TOpp'] = np.asarray([M.get_T0_ion_vol('O',2) for M in self.Ms])
        self.I1['TOpp_PN'] = O3.getTemDen(self.I1['O3as']/self.I1['O3s'], den=10, wave1 = 4363, wave2 = 5007)
        self.I1['Opp'] = np.asarray([M.get_ab_ion_vol_ne('O', 2)*10**M.abund['O'] for M in self.Ms])
        self.I1['Clpp'] = np.asarray([M.get_ab_ion_vol_ne('Cl', 2)*10**M.abund['Cl'] for M in self.Ms])
        self.I1['Opp_PN'] = O3.getIonAbundance(self.I1['O3s']/self.I1['Hbs'], tem=self.I1['TOpp_PN'], den=10 * np.ones_like(self.I1['O3s']), wave=5007, Hbeta=1)
        self.I1['TNp'] = np.asarray([M.get_T0_ion_vol('N',1) for M in self.Ms])
        self.I1['TNp_PN'] = N2.getTemDen(self.I1['N2as']/self.I1['N2s'], den=10, wave1 = 5755, wave2 = 6584)
        self.I1['Np'] = np.asarray([M.get_ab_ion_vol_ne('N', 1)*10**M.abund['N'] for M in self.Ms])
        self.I1['Np_PN'] = N2.getIonAbundance(self.I1['N2s']/self.I1['Hbs'], tem=self.I1['TNp_PN'], den=10 * np.ones_like(self.I1['N2s']), wave=6584, Hbeta=1)
        self.I1['Op'] = np.asarray([M.get_ab_ion_vol_ne('O', 1)*10**M.abund['O'] for M in self.Ms])
        self.I1['Op_PN'] = O2.getIonAbundance(self.I1['O2s']/self.I1['Hbs'], tem=self.I1['TNp_PN'], den=10 * np.ones_like(self.I1['O2s']), to_eval='I(2, 1) + I(3, 1)', Hbeta=1)

        self.I1['NpOp'] = np.asarray([M.get_ab_ion_vol_ne('N', 1) / M.get_ab_ion_vol_ne('O', 1) for M in self.Ms])
        self.I1['ClppOpp'] = np.asarray([M.get_ab_ion_vol_ne('Cl', 2) / M.get_ab_ion_vol_ne('O', 2) for M in self.Ms])


    def plot_I1(self, axes=None):
        I = self.I1
        if axes is None:
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,6))
        else:
            ax1, ax2, ax3 = axes
            f = plt.gcf()
        ax1.scatter(I['12OHs'], I['logUs'])
        ax1.set_xlabel(f'{l_dict["12OHs"]}')
        ax1.set_ylabel('log(U)')
        ax2.scatter(I['12OHs'], I['NOs'])
        ax2.set_xlabel(f'{l_dict["12OHs"]}')
        ax2.set_ylabel('log(N/O)')
        sc3 = ax3.scatter(np.log10(I['N2s']/I['Has']), np.log10(I['O3s']/I['Hbs']), c=12+I['OHs'], cmap='jet', s=100)
        ax3.set_xlabel(f'log({l_dict["N2s"]}/{l_dict["Has"]})')
        ax3.set_ylabel(f'log({l_dict["O3s"]}/{l_dict["Hbs"]})')
        x_tab = np.linspace(-3.5, 0, 100)
        ax3.plot(x_tab, Kewley(x_tab), c='b', label='Kewley+01')
        ax3.plot(x_tab, Kauffman(x_tab), c='k', label='Kauffman+03')
        ax3.plot(x_tab, Stasinska(x_tab), c='r', label='Stasinska+00')
        ax3.set_ylim(-2, 1.5)
        ax3.legend()
        cax = ax3.inset_axes((1.05, 0, 0.08, 1.0))
        cb3 = f.colorbar(sc3, cax=cax)
        cb3.set_label('12+log(O/H)')

        f.tight_layout()

    def plot_params(self, axes=None):
        I = self.I1
        if axes is None:
            f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7, 6))
        else:
            ax1, ax2, ax3, ax4 = axes
            f = plt.gcf()
        ax1.scatter(I['12OHs'], I['logUs'])
        ax1.set_xlabel(f'{l_dict["12OHs"]}')
        ax1.set_ylabel(f'{l_dict["logUs"]}')

        ax2.scatter(I['12OHs'], I['logQ0s'])
        ax2.set_xlabel(f'{l_dict["12OHs"]}')
        ax2.set_ylabel(f'{l_dict["logQ0s"]}')

        ax3.scatter(I['12OHs'], I['logRins'])
        ax3.set_xlabel(f'{l_dict["12OHs"]}')
        ax3.set_ylabel(f'{l_dict["logRins"]}')

        f.tight_layout()

    def set_2D(self, dims):

        self.M2Ds = [pc.C3D(M, dims=dims, n_dim=1, center=False) for M in self.Ms]
        self.Maps = {}
        self.Maps['Has'] = np.asarray([M2D.get_emis('H__1_656280A') for M2D in self.M2Ds])
        self.Maps['Hbs'] = np.asarray([M2D.get_emis('H__1_486132A') for M2D in self.M2Ds])
        self.Maps['N2s'] = np.asarray([M2D.get_emis('N__2_658345A') for M2D in self.M2Ds])
        self.Maps['N2as'] = np.asarray([M2D.get_emis('N__2_575459A') + M2D.get_emis('N_2R_575500A') for M2D in self.M2Ds])
        self.Maps['O2s'] = np.asarray([M2D.get_emis('O__2_372603A') + M2D.get_emis('O__2_372881A') for M2D in self.M2Ds])
        self.Maps['O3s'] = np.asarray([M2D.get_emis('O__3_500684A') for M2D in self.M2Ds])
        self.Maps['O3as'] = np.asarray([M2D.get_emis('O__3_436321A') + M2D.get_emis('O_3R_436300A') for M2D in self.M2Ds])
        self.Maps['R23s'] = np.asarray([1.3*M2D.get_emis('O__3_500684A') + M2D.get_emis('O__2_372603A') + M2D.get_emis('O__2_372881A') for M2D in self.M2Ds])
        self.Maps['S2s'] = np.asarray([(M2D.get_emis('S__2_671644A') + M2D.get_emis('S__2_673082A')) for M2D in self.M2Ds])
        self.Maps['S3s'] = np.asarray([M2D.get_emis('S__3_906862A') for M2D in self.M2Ds])
        self.Maps['Ar3s'] = np.asarray([M2D.get_emis('AR_3_713579A') for M2D in self.M2Ds])
        self.Maps['pR2s'] = self.Maps['O2s'] 
        self.Maps['pN2s'] = 1.33 * self.Maps['N2s'] 
        self.Maps['pS2s'] = self.Maps['S2s'] 
        self.Maps['pR3s'] = 1.33 * self.Maps['O3s'] 
        self.Maps['O1s'] = np.asarray([M2D.get_emis('O__1_630030A') for M2D in self.M2Ds])
        self.Maps['Cl3s'] = np.asarray([(M2D.get_emis('CL_3_551771A') + M2D.get_emis('CL_3_553787A')) for M2D in self.M2Ds])

        self.I2 = {}
        for l in self.Maps:
            self.Maps[l] = np.where(np.isfinite(self.Maps[l]), self.Maps[l], 0)
            self.I2[l] = self.Maps[l].sum(2)
        self.I2['N2S2Ha'] = 10**(np.log10(self.I2['N2s'] / self.I2['S2s']) + 0.264 * np.log10(self.I2['N2s'] / self.I2['Has']))
        self.I2['x'] = np.asarray([M2D.cub_coord.x_vec / np.max(M2D.cub_coord.x_vec) for M2D in self.M2Ds])
        self.I2['OHs'] = np.asarray([M2D.m[0].abund['O']*np.ones_like(M2D.cub_coord.x_vec) for M2D in self.M2Ds])
        self.I2['12OHs'] = 12 + self.I2['OHs']
        self.I2['TOpp'] = np.asarray([(M2D.get_ionic('O',2) * M2D.te).sum(1) / (M2D.get_ionic('O',2).sum(1)) for M2D in self.M2Ds])
        self.I2['TOpp_PN'] = O3.getTemDen(self.I2['O3as']/self.I2['O3s'], den=10, wave1 = 4363, wave2 = 5007)
        self.I2['Opp'] = np.asarray([((M2D.get_ionic('O',2)*M2D.ne).sum(1) / M2D.ne.sum(1)) for M2D in self.M2Ds]) * np.asarray([10**M.abund['O']* np.ones(self.dims) for M in self.Ms])
        self.I2['Opp_PN'] = O3.getIonAbundance(self.I2['O3s']/self.I2['Hbs'], tem=self.I2['TOpp_PN'], den=10 * np.ones_like(self.I2['x']), wave=5007, Hbeta=1)
        self.I2['Clpp'] = np.asarray([((M2D.get_ionic('Cl',2)*M2D.ne).sum(1) / M2D.ne.sum(1)) for M2D in self.M2Ds]) * np.asarray([10**M.abund['Cl']* np.ones(self.dims) for M in self.Ms])
        self.I2['Clpp_PN'] = Cl3.getIonAbundance(self.I2['Cl3s']/self.I2['Hbs'], tem=self.I2['TOpp_PN'], den=10 * np.ones_like(self.I2['x']),to_eval='I(2, 1) + I(3, 1)', Hbeta=1)
        self.I2['TNp'] = np.asarray([(M2D.get_ionic('N',1) * M2D.te).sum(1) / (M2D.get_ionic('N',1).sum(1)) for M2D in self.M2Ds])
        self.I2['TNp_PN'] = N2.getTemDen(self.I2['N2as']/self.I2['N2s'], den=10, wave1 = 5755, wave2 = 6584)
        self.I2['Np'] = np.asarray([((M2D.get_ionic('N',1)*M2D.ne).sum(1)/M2D.ne.sum(1)) for M2D in self.M2Ds]) * np.asarray([10**M.abund['N']* np.ones(self.dims) for M in self.Ms])
        self.I2['Np_PN'] = N2.getIonAbundance(self.I2['N2s']/self.I2['Hbs'], tem=self.I2['TNp_PN'], den=10 * np.ones_like(self.I2['x']), wave=6584, Hbeta=1)
        self.I2['Op'] = np.asarray([((M2D.get_ionic('O',1)*M2D.ne).sum(1)/M2D.ne.sum(1)) for M2D in self.M2Ds])  * np.asarray([10**M.abund['O']* np.ones(self.dims) for M in self.Ms])
        self.I2['Op_PN'] = O2.getIonAbundance(self.I2['O2s']/self.I2['Hbs'], tem=self.I2['TNp_PN'], den=10 * np.ones_like(self.I2['x']), to_eval='I(2, 1) + I(3, 1)', Hbeta=1)

        self.I2['NpOp'] = [((M2D.get_ionic('N',1)*M2D.ne).sum(1) / (M2D.get_ionic('O',1)*M2D.ne).sum(1)) for M2D in self.M2Ds]
        self.I2['ClppOpp'] = [((M2D.get_ionic('Cl',2)*M2D.ne).sum(1) / (M2D.get_ionic('O',2)*M2D.ne).sum(1)) for M2D in self.M2Ds]
        for pil_str in ('pR2s', 'pN2s', 'pS2s', 'pR3s'):
            self.I2[pil_str] = self.I2[pil_str] / self.I2['Hbs']
        self.I2['pOH_RU'] = (8.589 + 
                             0.022 * np.log10(self.I2['pR3s']/self.I2['pR2s']) + 
                             0.399 * np.log10(self.I2['pN2s']) + 
                             np.log10(self.I2['pR2s']) * 
                                (-0.137 + 0.164 * np.log10(self.I2['pR3s']/self.I2['pR2s']) + 0.589 * np.log10(self.I2['pN2s']))
                             )
        self.I2['pOH_RL'] = (7.932 +
                             0.944 * np.log10(self.I2['pR3s']/self.I2['pR2s']) + 
                             0.695 * np.log10(self.I2['pN2s']) + 
                             np.log10(self.I2['pR2s']) * 
                                (0.970 - 0.291 * np.log10(self.I2['pR3s']/self.I2['pR2s']) - 0.019 * np.log10(self.I2['pN2s']))
                            )
        self.I2['pOH_R'] = np.where(np.log10(self.I2['pR2s']) > -0.6, self.I2['pOH_RU'], self.I2['pOH_RL'])
        self.I2['pOH_SU'] = (8.424 +
                             0.030 * np.log10(self.I2['pR3s']/self.I2['pS2s']) + 
                             0.751 * np.log10(self.I2['pN2s']) + 
                             np.log10(self.I2['pS2s']) * 
                                (-0.349 + 0.182 * np.log10(self.I2['pR3s']/self.I2['pS2s']) + 0.508 * np.log10(self.I2['pN2s']))
                            )
        self.I2['pOH_SL'] = (8.072 +
                             0.789 * np.log10(self.I2['pR3s']/self.I2['pS2s']) + 
                             0.726 * np.log10(self.I2['pN2s']) + 
                             np.log10(self.I2['pR2s']) * 
                                (1.069 - 0.170 * np.log10(self.I2['pR3s']/self.I2['pS2s']) - 0.022 * np.log10(self.I2['pN2s']))
                            )
        self.I2['pOH_S'] = np.where(np.log10(self.I2['pN2s']) > -0.6, self.I2['pOH_SU'], self.I2['pOH_SL'])
    def set_mask(self, limit):
        masks = []
        for st in ('Has', 'Hbs', 'N2s', 'O3s', 'S2s'):#, 'O1s'):
            masks.append(self.I2[st] > np.tile(limit*self.I2[st].max(1), (self.I2[st].shape[1], 1)).T)
        self.mask = np.logical_and.reduce(masks)

    def plot_BPT(self, x1_str, x2_str, y1_str, y2_str, xlim, ylim, 
                 xlog=True, ylog=True, ax=None, plot_KKS=False, plot_cb = True,
                 vmin=None, vmax=None):

        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(8, 6))
        else:
            f = plt.gcf()
        I1 = self.I1
        I2 = self.I2
        #for i2 in self.I2:
        #    masks = []
        #    for mapl in ('Has', 'Hbs', 'O3s', 'N2s', 'O1s', 'S2s'):
        #        masks.append(i2[mapl] > 0.05 * i2[mapl].max())
        #    mask = np.logical_and.reduce(masks)
        #    ax.scatter(np.log10(i2['N2s'] / i2['Has'])[mask], np.log10(i2['O3s'] / i2['Hbs'])[mask], c=i2['OHs'][mask], s=400 ** i2['x'][mask], cmap='jet', vmin = -4.9, vmax = -3.1)  
        m = self.mask
        if x2_str is not None:
            x1 = I1[x1_str] / I1[x2_str]
            x2 = I2[x1_str] / I2[x2_str]
            x_label = f'{l_dict[x1_str]}/{l_dict[x2_str]}'
        else:
            x1 = I1[x1_str]
            x2 = I2[x1_str]
            x_label = f'{l_dict[x1_str]}'
        if y2_str is not None:
            y1 = I1[y1_str] / I1[y2_str]
            y2 = I2[y1_str] / I2[y2_str]
            y_label = f'{l_dict[y1_str]}/{l_dict[y2_str]}'
        else:
            y1 = I1[y1_str]
            y2 = I2[y1_str]
            y_label = f'{l_dict[y1_str]}'
        if xlog:
            x1 = np.log10(x1)
            x2 = np.log10(x2)
            x_label = f'log({x_label})'
        if ylog:
            y1 = np.log10(y1)
            y2 = np.log10(y2)
            y_label = f'log({y_label})'
        if vmin is None and vmax is None:
            if ylim is not None:
                vmin = ylim[0]
                vmax = ylim[1]
            else:
                vmin = -4.9+12
                vmax = -3.1+12
        ax.scatter(x2[m], y2[m], c=12+I2['OHs'][m], s=400 ** I2['x'][m], cmap='jet', vmin = vmin, vmax = vmax)
        sc = ax.scatter(x1, y1, cmap='jet',  c=12+I1['OHs'], s=250, marker='*', edgecolors='k', vmin = vmin, vmax=vmax)
        if plot_cb:
            cb = plt.colorbar(sc, ax=ax)
            cb.set_label('12 + log(O/H)')
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if plot_KKS:
            x_tab = np.linspace(xlim[0], xlim[1], 100)
            ax.plot(x_tab, Kewley(x_tab), c='b', label='Kewley')
            ax.plot(x_tab, Kauffman(x_tab), c='k', label='Kauffman')
            ax.plot(x_tab, Stasinska(x_tab), c='r', label='Stasinska')
            ax.legend()

# %%
