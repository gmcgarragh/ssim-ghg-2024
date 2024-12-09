import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import sys
import os
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import settings as s
from absco_lookup import sigma_lookup
from copy import deepcopy
import time
import xrtm

#Constants
g=9.81 #m/s^2
M=.0289644 #kg/mol
Na=6.022e23 #molecule/mol^-1
sigma_sun = 6.794e-5 #sr
h = 6.62607015e-34 #m^2*kg/s
c = 2.99792458e8 #m/s
k = 1.381e-23 #JK^-1molecule^-1

#Constants for Rayleigh scattering calculations
a_rayleigh = 2.871e-4
b_rayleigh = 5.67e-3
rho_rayleigh = 0.0279 #depolarization factor
epsilon = 0.622

#Other variables that we're hard-coding.
ILS_width = 5.0


class ForwardFunction:
    def __init__(self,SNR=s.SNR,sza_0=s.sza_0,sza=s.sza,co2=s.co2_true,ch4=s.ch4_true,T=s.T_true,p=s.p_true,q=s.q_true,albedo=s.albedo_true,band_min_wn=s.band_min_wn,band_max_wn=s.band_max_wn,band_spectral_resolutions=s.band_spectral_resolutions,band_min_um=s.band_min_um,band_max_um=s.band_max_um,band_spectral_points=s.band_spectral_points,band_wn=s.band_wn,band_wl=s.band_wl,band_absco_res_wn=s.band_absco_res_wn,resolving_power_band=s.resolving_power_band,sigma_band=s.sigma_band,band_wn_index=s.band_wn_index,ILS_Gaussian_term=s.ILS_Gaussian_term,ILS_Gaussian_term_sum=s.ILS_Gaussian_term_sum,absco_data=None,band_molecules=s.band_molecules,P_aerosol=s.P_aerosol,ssa_aerosol=s.ssa_aerosol,qext_aerosol=s.qext_aerosol,height_aerosol=s.height_aerosol,tau_aerosol=None,measurement_error=False,jacobians=False):


        self.SNR = SNR
        self.sza_0 = sza_0
        self.sza = sza
        self.co2 = co2
        self.ch4 = ch4
        self.T = T
        self.p = p
        self.q = q
        self.albedo = albedo
        self.band_min_wn = band_min_wn
        self.band_max_wn = band_max_wn
        self.band_spectral_resolutions = band_spectral_resolutions
        self.band_molecules = band_molecules
        self.P_aerosol = P_aerosol
        self.ssa_aerosol = ssa_aerosol
        self.qext_aerosol = qext_aerosol
        self.height_aerosol = height_aerosol
        self.tau_aerosol = tau_aerosol
        self.measurement_error = measurement_error
        self.jacobians = jacobians
        self.band_min_um = band_min_um
        self.band_max_um = band_max_um
        self.band_spectral_points = band_spectral_points
        self.band_wn = band_wn
        self.band_wl = band_wl
        self.band_absco_res_wn = band_absco_res_wn
        self.resolving_power_band = resolving_power_band
        self.sigma_band = sigma_band
        self.band_wn_index = band_wn_index
        self.ILS_Gaussian_term = ILS_Gaussian_term
        self.ILS_Gaussian_term_sum = ILS_Gaussian_term_sum

        #Approximately calculate the solar irradiance at our bands using Planck's law (assuming the Sun is 5800 K), then account for the solid angle of the Sun and convert into per um instead of per m.
        self.band_solar_irradiances = np.empty((len(self.band_min_wn)))
        for i in range(len(self.band_max_wn)):
            self.band_solar_irradiances[i] = planck(5800.,np.mean(self.band_wl[i])*1e-6)*sigma_sun/1.e6 #W/m^2/um

        #Geometry
        self.mu_0=np.cos(np.deg2rad(self.sza_0)) #cosine of the solar zenith angle [deg]
        self.mu=np.cos(np.deg2rad(self.sza)) #cosine of the sensor zenith angle [deg]
        self.m = 1/self.mu_0+1/self.mu #airmass

        #Calculate the d(pressure) of a given layer
        self.p_diff=np.empty((len(self.p)-1))
        for i in range(len(self.p)-1):
            self.p_diff[i] = self.p[i+1]-self.p[i]

        #Layer calculations
        self.co2_layer = layer_average(self.co2)
        self.ch4_layer = layer_average(self.ch4)
        self.T_layer = layer_average(self.T)
        self.p_layer = layer_average(self.p)
        self.q_layer = layer_average(self.q)

        #Calculate the true Xgases from the gas profiles
        self.xco2, self.h = calculate_Xgas(self.co2, self.p, self.q)
        self.xch4, self.h = calculate_Xgas(self.ch4, self.p, self.q)

        self.xco2 *= 1e6 #Convert from mol/mol to ppm
        self.xch4 *= 1e9 #Convert from mol/mol to ppb

        ##########################################################
        #Calculate absorption cross sections for H2O, O2, and CO2 in m^2/molecule, as needed (i.e., there's no CO2 absorption in the OCO-2 O2 A-band)
        self.tau_star_band = [] #band_n total optical depths
        self.tau_above_aerosol_star_band = [] #band_n total optical depths above the aerosol layer

        #For analytical Jacobians. Calculate on each layer.
        self.tau_star_band_q = []
        self.tau_above_aerosol_star_band_q = []
        self.tau_star_band_co2 = []
        self.tau_above_aerosol_star_band_co2 = []
        self.tau_star_band_ch4 = []
        self.tau_above_aerosol_star_band_ch4 = []

        #Loop through the bands
        for i in range(len(self.band_min_um)):

            #Loop through the desired molecules for this band
            tau_star_temp = np.zeros((len(self.band_absco_res_wn[i])))
            tau_above_aerosol_star_temp = np.zeros((len(self.band_absco_res_wn[i])))

            #For analytical Jacobians (need q, co2, ch4)
            tau_star_temp_q = np.zeros((len(self.band_absco_res_wn[i]),len(self.p_layer)))
            tau_above_aerosol_star_temp_q = np.zeros((len(self.band_absco_res_wn[i]),len(self.p_layer)))
            tau_star_temp_co2 = np.zeros((len(self.band_absco_res_wn[i]),len(self.p_layer)))
            tau_above_aerosol_star_temp_co2 = np.zeros((len(self.band_absco_res_wn[i]),len(self.p_layer)))
            tau_star_temp_ch4 = np.zeros((len(self.band_absco_res_wn[i]),len(self.p_layer)))
            tau_above_aerosol_star_temp_ch4 = np.zeros((len(self.band_absco_res_wn[i]),len(self.p_layer)))

            for j in range(len(self.band_molecules[i])):
                molecules_sigma_temp = sigma_lookup(self.band_molecules[i][j],self.band_absco_res_wn[i],self.p_layer,self.T_layer,absco_data)

                #Tau is the cross section times number density times dz. We can assume hydrostatic equilibrium and the ideal gas law to solve it using this equation instead:
                if self.band_molecules[i][j] == 'o2':
                    tau_temp = np.tile(self.p_diff,(len(self.band_absco_res_wn[i]),1)) * 0.20935 * molecules_sigma_temp / M / g
                    if self.jacobians:
                        tau_temp_q = np.zeros((len(self.band_absco_res_wn[i]),len(self.p_diff)))
                        tau_temp_co2 = np.zeros((len(self.band_absco_res_wn[i]),len(self.p_diff)))
                        tau_temp_ch4 = np.zeros((len(self.band_absco_res_wn[i]),len(self.p_diff)))

                elif self.band_molecules[i][j] == 'h2o':
                    tau_temp = np.tile(self.p_diff,(len(self.band_absco_res_wn[i]),1)) * np.tile(self.q_layer,(len(molecules_sigma_temp),1)) * molecules_sigma_temp / M / g
                    if self.jacobians:
                        tau_temp_q = np.tile(self.p_diff,(len(self.band_absco_res_wn[i]),1)) * molecules_sigma_temp / M / g
                        tau_temp_co2 = np.zeros((len(self.band_absco_res_wn[i]),len(self.p_diff)))
                        tau_temp_ch4 = np.zeros((len(self.band_absco_res_wn[i]),len(self.p_diff)))

                elif self.band_molecules[i][j] == 'co2':
                    tau_temp = np.tile(self.p_diff,(len(self.band_absco_res_wn[i]),1)) * np.tile(self.co2_layer,(len(molecules_sigma_temp),1)) * molecules_sigma_temp / M / g
                    if self.jacobians:
                        tau_temp_q = np.zeros((len(self.band_absco_res_wn[i]),len(self.p_diff)))
                        tau_temp_co2 = np.tile(self.p_diff,(len(self.band_absco_res_wn[i]),1)) * molecules_sigma_temp / M / g
                        tau_temp_ch4 = np.zeros((len(self.band_absco_res_wn[i]),len(self.p_diff)))

                elif self.band_molecules[i][j] == 'ch4':
                    tau_temp = np.tile(self.p_diff,(len(self.band_absco_res_wn[i]),1)) * np.tile(self.ch4_layer,(len(molecules_sigma_temp),1)) * molecules_sigma_temp / M / g
                    if self.jacobians:
                        tau_temp_q = np.zeros((len(self.band_absco_res_wn[i]),len(self.p_diff)))
                        tau_temp_co2 = np.zeros((len(self.band_absco_res_wn[i]),len(self.p_diff)))
                        tau_temp_ch4 = np.tile(self.p_diff,(len(self.band_absco_res_wn[i]),1)) * molecules_sigma_temp / M / g

                else:
                    print("Choose a valid molecule!")
                    return

                #Sum the vertical profile dimension
                tau_star_temp += np.sum(tau_temp,axis=1)

                #Sum the vertical profile dimension but only above the aerosol layer
                tau_above_aerosol_star_temp += np.sum(tau_temp[:,self.p_layer < self.height_aerosol],axis=1)

                if self.jacobians:
                    #For analytic Jacobians, save each layer
                    tau_star_temp_q += tau_temp_q
                    tau_star_temp_co2 += tau_temp_co2
                    tau_star_temp_ch4 += tau_temp_ch4

                    #For analytic Jacobians, only add the layer aod above height_aerosol
                    tau_above_aerosol_star_temp_q[:,self.p_layer < self.height_aerosol] += tau_temp_q[:,self.p_layer < self.height_aerosol]
                    tau_above_aerosol_star_temp_co2[:,self.p_layer < self.height_aerosol] += tau_temp_co2[:,self.p_layer < self.height_aerosol]
                    tau_above_aerosol_star_temp_ch4[:,self.p_layer < self.height_aerosol] += tau_temp_ch4[:,self.p_layer < self.height_aerosol]


            #Also add the Rayleigh scattering optical depth
            tau_rayleigh_band = np.empty((len(self.band_absco_res_wn[i]),len(self.p_layer)))

            #See Section 3.2.1.5 in the OCO-2 L2 ATBD
            n_s = 1 + a_rayleigh*(1. + b_rayleigh*np.mean(self.band_wl[i])**2.) #Just use Rayleigh in the middle of the O2 A-band (in um here)

            #Typo in the ATBD. Ns should be e-24 instead of e-20.
            rayleigh_sigma_band = (1.031e-24 * (n_s**2. - 1.)**2.)/(np.mean(self.band_wl[i])**4. * (n_s**2. + 2.)**2)*(6.+3.*rho_rayleigh)/(6.-7.*rho_rayleigh)
            #Simplify
            tau_rayleigh_band = self.p_diff * Na * rayleigh_sigma_band / M / g

            tau_star_temp += np.sum(tau_rayleigh_band)
            tau_above_aerosol_star_temp += np.sum(tau_rayleigh_band[self.p_layer < self.height_aerosol])

            #Append for the band we're on
            self.tau_star_band.append(tau_star_temp)
            self.tau_above_aerosol_star_band.append(tau_above_aerosol_star_temp)

            #For analytic Jacobians
            self.tau_star_band_q.append(tau_star_temp_q)
            self.tau_above_aerosol_star_band_q.append(tau_above_aerosol_star_temp_q)
            self.tau_star_band_co2.append(tau_star_temp_co2)
            self.tau_above_aerosol_star_band_co2.append(tau_above_aerosol_star_temp_co2)
            self.tau_star_band_ch4.append(tau_star_temp_ch4)
            self.tau_above_aerosol_star_band_ch4.append(tau_above_aerosol_star_temp_ch4)


        #####
        #Calculate radiances for each band
        self.R_band = []
        self.R_band_albedo = []
        self.R_band_aerosol = []
        self.R_band_q = []
        self.R_band_co2 = []
        self.R_band_ch4 = []

        #print("Calculating radiances...")
        for i in range(len(self.band_min_um)):

            if self.tau_aerosol != None:
              tau_aerosol_temp = np.full(len(self.band_absco_res_wn[i]),self.tau_aerosol)
            else:
              tau_aerosol_temp = np.zeros(len(self.band_absco_res_wn[i]))

            I, I_albedo, I_aerosol, I_q, I_co2, I_ch4 = self.intensity(
                self.band_absco_res_wn[i],
                self.tau_star_band[i],
                self.tau_above_aerosol_star_band[i],
                self.tau_star_band_q[i],
                self.tau_above_aerosol_star_band_q[i],
                self.tau_star_band_co2[i],
                self.tau_above_aerosol_star_band_co2[i],
                self.tau_star_band_ch4[i],
                self.tau_above_aerosol_star_band_ch4[i],
                tau_aerosol_temp,
                self.ssa_aerosol[i],
                self.P_aerosol[i],
                self.qext_aerosol[0],
                self.qext_aerosol[i],
                self.mu,
                self.mu_0,
                self.m,
                self.albedo[i],
                self.band_solar_irradiances[i],
                self.jacobians)

            #Calculate the spectral response function (with and without multiplying by intensity)
            Sc_I_band, Sc_I_band_albedo, Sc_I_band_aerosol, Sc_I_band_q, Sc_I_band_co2, Sc_I_band_ch4 = self.spectral_response_function(self.band_wn_index[i],self.band_absco_res_wn[i],self.sigma_band[i],self.ILS_Gaussian_term[i],I,I_albedo,I_aerosol,I_q,I_co2,I_ch4,self.jacobians)

            #Calculate radiance (Rc) by integrating intensity times ILS, and reverse to plot in micrometers
            Rc_band = (Sc_I_band/self.ILS_Gaussian_term_sum[i])[::-1]

            #For analytic Jacobians
            if jacobians:
              Rc_band_albedo = (Sc_I_band_albedo/self.ILS_Gaussian_term_sum[i])[::-1]
              Rc_band_aerosol = (Sc_I_band_aerosol/self.ILS_Gaussian_term_sum[i])[::-1]
              Rc_band_q = (Sc_I_band_q/self.ILS_Gaussian_term_sum[i][:,None])[::-1,:]
              Rc_band_co2 = (Sc_I_band_co2/self.ILS_Gaussian_term_sum[i][:,None])[::-1,:]
              Rc_band_ch4 = (Sc_I_band_ch4/self.ILS_Gaussian_term_sum[i][:,None])[::-1,:]

            #Append for the band we're on
            self.R_band.append(Rc_band)
            if jacobians:
              #For analytic Jacobians
              self.R_band_albedo.append(Rc_band_albedo)
              self.R_band_aerosol.append(Rc_band_aerosol)
              self.R_band_q.append(Rc_band_q)
              self.R_band_co2.append(Rc_band_co2)
              self.R_band_ch4.append(Rc_band_ch4)

        self.y = np.concatenate(self.R_band)
        if jacobians:
          self.y_albedo = np.concatenate(self.R_band_albedo)
          self.y_aerosol = np.concatenate(self.R_band_aerosol)
          self.y_q = np.concatenate(self.R_band_q)
          self.y_co2 = np.concatenate(self.R_band_co2)
          self.y_ch4 = np.concatenate(self.R_band_ch4)

        noise = []
        for i in range(len(self.band_max_wn)):
          signal = self.R_band[i].max()
          sigma = signal/self.SNR
          np.random.seed(0)
          noise_temp = np.random.normal(0,sigma,self.band_spectral_points[i])
          noise.append(noise_temp)

          #If we're adding noise:
          if self.measurement_error: self.R_band[i] = self.R_band[i] + noise_temp

        #Now combine into one spectra
        self.y = np.concatenate(self.R_band)

        #Calculate Sy even if we didn't add noise because we need something for Sy
        noise_std = []
        for i in range(len(self.band_max_wn)):
          noise_std.append(np.full((len(noise[i])),np.std(noise[i])**2.0))
        Sy = np.diag(np.concatenate(noise_std))

        #Calculate this ahead of time so we don't have to calculate it every retrieval iteration
        Sy_inv = np.zeros(Sy.shape)
        np.fill_diagonal(Sy_inv,1./Sy.diagonal())

        self.Sy_inv = Sy_inv


    #Calculate intensities for a single band
    def intensity(self,
                  band,
                  tau_star_band,
                  tau_above_aerosol_star_band,
                  tau_star_band_q,
                  tau_above_aerosol_star_band_q,
                  tau_star_band_co2,
                  tau_above_aerosol_star_band_co2,
                  tau_star_band_ch4,
                  tau_above_aerosol_star_band_ch4,
                  tau_aerosol,
                  ssa_aerosol,
                  P_aerosol,
                  qext_aerosol_band_0,
                  qext_aerosol,
                  mu,
                  mu_0,
                  m,
                  albedo,
                  band_solar_irradiances,
                  jacobians):

      I = np.zeros((len(band))) #wn
      I_albedo = np.zeros((len(band))) #wn
      I_aerosol = np.zeros((len(band))) #wn
      I_q = np.zeros((len(band),tau_star_band_q.shape[1])) #wn x layers
      I_co2 = np.zeros((len(band),tau_star_band_q.shape[1])) #wn x layers
      I_ch4 = np.zeros((len(band),tau_star_band_q.shape[1])) #wn x layers

      #Dealing with divide by zero issues
      if qext_aerosol_band_0[0] == 0:
        qext_scaling = np.zeros((len(qext_aerosol)))
      else:
        qext_scaling = qext_aerosol/qext_aerosol_band_0[0]

      #Direct exponential term
      exp_term = np.exp(-m*(tau_star_band + tau_aerosol*qext_scaling))

      #Scattering exponential term
      exp_term_above_aerosol = np.exp(-m*tau_above_aerosol_star_band)

      for i in range(len(band)):
        #Add an aerosol layer. Assume it scatters once.
        #Full qext scaling
        I[i] = band_solar_irradiances/np.pi * (albedo*mu_0*exp_term[i] + ssa_aerosol[i]*P_aerosol[i]*tau_aerosol[i]*qext_scaling[i]*exp_term_above_aerosol[i]/4./mu)

        #Calculate analytical Jacobians
        if jacobians:
          I_albedo[i] = band_solar_irradiances/np.pi * mu_0 * exp_term[i]

          I_aerosol[i] = band_solar_irradiances/np.pi * (-m*qext_scaling[i]*albedo*mu_0*exp_term[i] + ssa_aerosol[i]*P_aerosol[i]*qext_scaling[i]*exp_term_above_aerosol[i]/4./mu)

          #Full qext scaling:
          I_q[i,:] = band_solar_irradiances/np.pi * (albedo*mu_0*exp_term[i] * (-m) * tau_star_band_q[i,:] + ssa_aerosol[i]*P_aerosol[i]*tau_aerosol[i]*qext_scaling[i]*exp_term_above_aerosol[i]/4./mu * (-m) * tau_above_aerosol_star_band_q[i,:])
          I_co2[i,:] = band_solar_irradiances/np.pi * (-m*(albedo*mu_0*exp_term[i] * tau_star_band_co2[i,:] + ssa_aerosol[i]*P_aerosol[i]*tau_aerosol[i]*qext_scaling[i]*exp_term_above_aerosol[i]/4./mu * tau_above_aerosol_star_band_co2[i,:]))
          I_ch4[i,:] = band_solar_irradiances/np.pi * (-m*(albedo*mu_0*exp_term[i] * tau_star_band_ch4[i,:] + ssa_aerosol[i]*P_aerosol[i]*tau_aerosol[i]*qext_scaling[i]*exp_term_above_aerosol[i]/4./mu * tau_above_aerosol_star_band_ch4[i,:]))

      return I, I_albedo, I_aerosol, I_q, I_co2, I_ch4


    #Assume a Gaussian ILS
    def spectral_response_function(self,band_wn_index,band,sigma_band,ILS_Gaussian_term,I_band,I_band_albedo,I_band_aerosol,I_band_q,I_band_co2,I_band_ch4,jacobians):

        Sc_I_band = np.zeros((len(band_wn_index))) #wn instrument

        Sc_I_band_albedo = np.zeros((len(band_wn_index))) #wn instrument
        Sc_I_band_aerosol = np.zeros((len(band_wn_index))) #wn instrument
        Sc_I_band_q = np.zeros((len(band_wn_index),I_band_q.shape[1])) #wn instrument x layers
        Sc_I_band_co2 = np.zeros((len(band_wn_index),I_band_co2.shape[1])) #wn instrument x layers
        Sc_I_band_ch4 = np.zeros((len(band_wn_index),I_band_ch4.shape[1])) #wn instrument x layers

        round_term = round(sigma_band*ILS_width*100.0)

        for i in range(len(band_wn_index)):

            #Dealing with the starting edge of the band
            if band[band_wn_index[i]] <= band[0]+sigma_band*ILS_width:

                j_index_temp_lower = 0
                j_index_temp_upper = int(band_wn_index[i]+round_term)

            #Dealing with the trailing edge of the band
            elif band[band_wn_index[i]] >= band[len(band)-1]-sigma_band*ILS_width:

                j_index_temp_lower = int(band_wn_index[i]-round_term)
                j_index_temp_upper = len(band)

            #Most of the band
            else:
                j_index_temp_lower = int(band_wn_index[i]-round_term)
                j_index_temp_upper = int(band_wn_index[i]+round_term)

            Sc_I_band[i] = np.sum(I_band[j_index_temp_lower:j_index_temp_upper] * ILS_Gaussian_term[i,j_index_temp_lower:j_index_temp_upper])

            if jacobians:
                Sc_I_band_albedo[i] = np.sum(I_band_albedo[j_index_temp_lower:j_index_temp_upper] * ILS_Gaussian_term[i,j_index_temp_lower:j_index_temp_upper])
                Sc_I_band_aerosol[i] = np.sum(I_band_aerosol[j_index_temp_lower:j_index_temp_upper] * ILS_Gaussian_term[i,j_index_temp_lower:j_index_temp_upper])
                Sc_I_band_q[i,:] = np.sum(I_band_q[j_index_temp_lower:j_index_temp_upper,:] * ILS_Gaussian_term[i,j_index_temp_lower:j_index_temp_upper,None],axis=0)
                Sc_I_band_co2[i,:] = np.sum(I_band_co2[j_index_temp_lower:j_index_temp_upper,:] * ILS_Gaussian_term[i,j_index_temp_lower:j_index_temp_upper,None],axis=0)
                Sc_I_band_ch4[i,:] = np.sum(I_band_ch4[j_index_temp_lower:j_index_temp_upper,:] * ILS_Gaussian_term[i,j_index_temp_lower:j_index_temp_upper,None],axis=0)

        return Sc_I_band, Sc_I_band_albedo, Sc_I_band_aerosol, Sc_I_band_q, Sc_I_band_co2, Sc_I_band_ch4


class ForwardFunction_xrtm:
    def __init__(self,SNR=s.SNR,sza_0=s.sza_0,sza=s.sza,phi=s.phi,co2=s.co2_true,ch4=s.ch4_true,T=s.T_true,p=s.p_true,q=s.q_true,albedo=s.albedo_true,band_min_wn=s.band_min_wn,band_max_wn=s.band_max_wn,band_spectral_resolutions=s.band_spectral_resolutions,band_min_um=s.band_min_um,band_max_um=s.band_max_um,band_spectral_points=s.band_spectral_points,band_wn=s.band_wn,band_wl=s.band_wl,band_absco_res_wn=s.band_absco_res_wn,resolving_power_band=s.resolving_power_band,sigma_band=s.sigma_band,band_wn_index=s.band_wn_index,ILS_Gaussian_term=s.ILS_Gaussian_term,ILS_Gaussian_term_sum=s.ILS_Gaussian_term_sum,absco_data=None,band_molecules=s.band_molecules,P_aerosol=s.P_aerosol,ssa_aerosol=s.ssa_aerosol,qext_aerosol=s.qext_aerosol,height_aerosol=s.height_aerosol,tau_aerosol=None,measurement_error=False,jacobians=False):


        self.SNR = SNR
        self.sza_0 = sza_0
        self.phi = phi
        self.sza = sza
        self.co2 = co2
        self.ch4 = ch4
        self.T = T
        self.p = p
        self.q = q
        self.albedo = albedo
        self.band_min_wn = band_min_wn
        self.band_max_wn = band_max_wn
        self.band_spectral_resolutions = band_spectral_resolutions
        self.band_molecules = band_molecules
        self.P_aerosol = P_aerosol
        self.ssa_aerosol = ssa_aerosol
        self.qext_aerosol = qext_aerosol
        self.height_aerosol = height_aerosol
        self.tau_aerosol = tau_aerosol
        self.measurement_error = measurement_error
        self.jacobians = jacobians
        self.band_min_um = band_min_um
        self.band_max_um = band_max_um
        self.band_spectral_points = band_spectral_points
        self.band_wn = band_wn
        self.band_wl = band_wl
        self.band_absco_res_wn = band_absco_res_wn
        self.resolving_power_band = resolving_power_band
        self.sigma_band = sigma_band
        self.band_wn_index = band_wn_index
        self.ILS_Gaussian_term = ILS_Gaussian_term
        self.ILS_Gaussian_term_sum = ILS_Gaussian_term_sum

        #Approximately calculate the solar irradiance at our bands using Planck's law (assuming the Sun is 5800 K), then account for the solid angle of the Sun and convert into per um instead of per m.
        self.band_solar_irradiances = np.empty((len(self.band_min_wn)))
        for i in range(len(self.band_max_wn)):
            self.band_solar_irradiances[i] = planck(5800.,np.mean(self.band_wl[i])*1e-6)*sigma_sun/1.e6 #W/m^2/um

        #Geometry
        self.mu_0=np.cos(np.deg2rad(self.sza_0)) #cosine of the solar zenith angle [deg]
        self.mu=np.cos(np.deg2rad(self.sza)) #cosine of the sensor zenith angle [deg]
        self.m = 1/self.mu_0+1/self.mu #airmass

        #Calculate the d(pressure) of a given layer
        self.p_diff=np.empty((len(self.p)-1))
        for i in range(len(self.p)-1):
            self.p_diff[i] = self.p[i+1]-self.p[i]

        #Layer calculations
        self.co2_layer = layer_average(self.co2)
        self.ch4_layer = layer_average(self.ch4)
        self.T_layer = layer_average(self.T)
        self.p_layer = layer_average(self.p)
        self.q_layer = layer_average(self.q)

        #Calculate the true Xgases from the gas profiles
        self.xco2, self.h = calculate_Xgas(self.co2, self.p, self.q)
        self.xch4, self.h = calculate_Xgas(self.ch4, self.p, self.q)

        self.xco2 *= 1e6 #Convert from mol/mol to ppm
        self.xch4 *= 1e9 #Convert from mol/mol to ppb

        ##########################################################
        #Calculate absorption cross sections for H2O, O2, and CO2 in m^2/molecule, as needed (i.e., there's no CO2 absorption in the OCO-2 O2 A-band)
        self.tau_star_band = [] #band_n total optical depths
        self.tau_above_aerosol_star_band = [] #band_n total optical depths above the aerosol layer

        #For analytical Jacobians. Calculate on each layer.
        self.tau_star_band_q = []
        self.tau_above_aerosol_star_band_q = []
        self.tau_star_band_co2 = []
        self.tau_above_aerosol_star_band_co2 = []
        self.tau_star_band_ch4 = []
        self.tau_above_aerosol_star_band_ch4 = []

        self.tau_layer_gas_band = [] #band_n layer optical depths
        self.tau_layer_ray_band = [] #band_n layer optical depths

        #Loop through the bands
        for i in range(len(self.band_min_um)):

            #Loop through the desired molecules for this band
            tau_star_temp = np.zeros((len(self.band_absco_res_wn[i])))
            tau_above_aerosol_star_temp = np.zeros((len(self.band_absco_res_wn[i])))

            tau_layer_gas_temp = np.zeros((len(self.band_absco_res_wn[i]),len(self.p_layer)))
            tau_layer_ray_temp = np.zeros((len(self.band_absco_res_wn[i]),len(self.p_layer)))

            #For analytical Jacobians (need q, co2, ch4)
            tau_star_temp_q = np.zeros((len(self.band_absco_res_wn[i]),len(self.p_layer)))
            tau_above_aerosol_star_temp_q = np.zeros((len(self.band_absco_res_wn[i]),len(self.p_layer)))
            tau_star_temp_co2 = np.zeros((len(self.band_absco_res_wn[i]),len(self.p_layer)))
            tau_above_aerosol_star_temp_co2 = np.zeros((len(self.band_absco_res_wn[i]),len(self.p_layer)))
            tau_star_temp_ch4 = np.zeros((len(self.band_absco_res_wn[i]),len(self.p_layer)))
            tau_above_aerosol_star_temp_ch4 = np.zeros((len(self.band_absco_res_wn[i]),len(self.p_layer)))

            for j in range(len(self.band_molecules[i])):
                molecules_sigma_temp = sigma_lookup(self.band_molecules[i][j],self.band_absco_res_wn[i],self.p_layer,self.T_layer,absco_data)

                #Tau is the cross section times number density times dz. We can assume hydrostatic equilibrium and the ideal gas law to solve it using this equation instead:
                if self.band_molecules[i][j] == 'o2':
                    tau_temp = np.tile(self.p_diff,(len(self.band_absco_res_wn[i]),1)) * 0.20935 * molecules_sigma_temp / M / g
                    if self.jacobians:
                        tau_temp_q = np.zeros((len(self.band_absco_res_wn[i]),len(self.p_diff)))
                        tau_temp_co2 = np.zeros((len(self.band_absco_res_wn[i]),len(self.p_diff)))
                        tau_temp_ch4 = np.zeros((len(self.band_absco_res_wn[i]),len(self.p_diff)))

                elif self.band_molecules[i][j] == 'h2o':
                    tau_temp = np.tile(self.p_diff,(len(self.band_absco_res_wn[i]),1)) * np.tile(self.q_layer,(len(molecules_sigma_temp),1)) * molecules_sigma_temp / M / g
                    if self.jacobians:
                        tau_temp_q = np.tile(self.p_diff,(len(self.band_absco_res_wn[i]),1)) * molecules_sigma_temp / M / g
                        tau_temp_co2 = np.zeros((len(self.band_absco_res_wn[i]),len(self.p_diff)))
                        tau_temp_ch4 = np.zeros((len(self.band_absco_res_wn[i]),len(self.p_diff)))

                elif self.band_molecules[i][j] == 'co2':
                    tau_temp = np.tile(self.p_diff,(len(self.band_absco_res_wn[i]),1)) * np.tile(self.co2_layer,(len(molecules_sigma_temp),1)) * molecules_sigma_temp / M / g
                    if self.jacobians:
                        tau_temp_q = np.zeros((len(self.band_absco_res_wn[i]),len(self.p_diff)))
                        tau_temp_co2 = np.tile(self.p_diff,(len(self.band_absco_res_wn[i]),1)) * molecules_sigma_temp / M / g
                        tau_temp_ch4 = np.zeros((len(self.band_absco_res_wn[i]),len(self.p_diff)))

                elif self.band_molecules[i][j] == 'ch4':
                    tau_temp = np.tile(self.p_diff,(len(self.band_absco_res_wn[i]),1)) * np.tile(self.ch4_layer,(len(molecules_sigma_temp),1)) * molecules_sigma_temp / M / g
                    if self.jacobians:
                        tau_temp_q = np.zeros((len(self.band_absco_res_wn[i]),len(self.p_diff)))
                        tau_temp_co2 = np.zeros((len(self.band_absco_res_wn[i]),len(self.p_diff)))
                        tau_temp_ch4 = np.tile(self.p_diff,(len(self.band_absco_res_wn[i]),1)) * molecules_sigma_temp / M / g

                else:
                    print("Choose a valid molecule!")
                    return

                #Sum the vertical profile dimension and accumulate
                tau_star_temp += np.sum(tau_temp,axis=1)

                #Accumulate for each layer and each molecule for XRTM
                tau_layer_gas_temp += tau_temp

                #Sum the vertical profile dimension but only above the aerosol layer
                tau_above_aerosol_star_temp += np.sum(tau_temp[:,self.p_layer < self.height_aerosol],axis=1)

                if self.jacobians:
                    #For analytic Jacobians, save each layer
                    tau_star_temp_q += tau_temp_q
                    tau_star_temp_co2 += tau_temp_co2
                    tau_star_temp_ch4 += tau_temp_ch4

                    #For analytic Jacobians, only add the layer aod above height_aerosol
                    tau_above_aerosol_star_temp_q[:,self.p_layer < self.height_aerosol] += tau_temp_q[:,self.p_layer < self.height_aerosol]
                    tau_above_aerosol_star_temp_co2[:,self.p_layer < self.height_aerosol] += tau_temp_co2[:,self.p_layer < self.height_aerosol]
                    tau_above_aerosol_star_temp_ch4[:,self.p_layer < self.height_aerosol] += tau_temp_ch4[:,self.p_layer < self.height_aerosol]


            #Also add the Rayleigh scattering optical depth
            tau_rayleigh_band = np.empty((len(self.band_absco_res_wn[i]),len(self.p_layer)))

            #See Section 3.2.1.5 in the OCO-2 L2 ATBD
            n_s = 1 + a_rayleigh*(1. + b_rayleigh*np.mean(self.band_wl[i])**2.) #Just use Rayleigh in the middle of the O2 A-band (in um here)

            #Typo in the ATBD. Ns should be e-24 instead of e-20.
            rayleigh_sigma_band = (1.031e-24 * (n_s**2. - 1.)**2.)/(np.mean(self.band_wl[i])**4. * (n_s**2. + 2.)**2)*(6.+3.*rho_rayleigh)/(6.-7.*rho_rayleigh)
            #Simplify
            tau_rayleigh_band = self.p_diff * Na * rayleigh_sigma_band / M / g

            tau_star_temp += np.sum(tau_rayleigh_band)

            tau_layer_ray_temp += tau_rayleigh_band

            tau_above_aerosol_star_temp += np.sum(tau_rayleigh_band[self.p_layer < self.height_aerosol])

            #Append for the band we're on
            self.tau_star_band.append(tau_star_temp)
            self.tau_above_aerosol_star_band.append(tau_above_aerosol_star_temp)

            self.tau_layer_gas_band.append(tau_layer_gas_temp)
            self.tau_layer_ray_band.append(tau_layer_ray_temp)

            #For analytic Jacobians
            self.tau_star_band_q.append(tau_star_temp_q)
            self.tau_above_aerosol_star_band_q.append(tau_above_aerosol_star_temp_q)
            self.tau_star_band_co2.append(tau_star_temp_co2)
            self.tau_above_aerosol_star_band_co2.append(tau_above_aerosol_star_temp_co2)
            self.tau_star_band_ch4.append(tau_star_temp_ch4)
            self.tau_above_aerosol_star_band_ch4.append(tau_above_aerosol_star_temp_ch4)


        #####
        #Calculate radiances for each band
        self.R_band = []
        self.R_band_albedo = []
        self.R_band_aerosol = []
        self.R_band_q = []
        self.R_band_co2 = []
        self.R_band_ch4 = []

        #print("Calculating radiances...")
        for i in range(len(self.band_min_um)):

            if self.tau_aerosol != None:
              tau_aerosol_temp = np.full(len(self.band_absco_res_wn[i]),self.tau_aerosol)
            else:
              tau_aerosol_temp = np.zeros(len(self.band_absco_res_wn[i]))

            I, I_albedo, I_aerosol, I_q, I_co2, I_ch4 = self.intensity_xrtm(
                i,
                self.tau_layer_gas_band[i],
                self.tau_layer_ray_band[i],
                self.p_layer > self.height_aerosol,
                tau_aerosol_temp,
                ssa_aerosol[i],
                P_aerosol[i],
                self.qext_aerosol[0],
                self.qext_aerosol[i],
                self.albedo[i],
                self.band_solar_irradiances[i])

            #Calculate the spectral response function (with and without multiplying by intensity)
            Sc_I_band, Sc_I_band_albedo, Sc_I_band_aerosol, Sc_I_band_q, Sc_I_band_co2, Sc_I_band_ch4 = self.spectral_response_function(self.band_wn_index[i],self.band_absco_res_wn[i],self.sigma_band[i],self.ILS_Gaussian_term[i],I,I_albedo,I_aerosol,I_q,I_co2,I_ch4,self.jacobians)

            #Calculate radiance (Rc) by integrating intensity times ILS, and reverse to plot in micrometers
            Rc_band = (Sc_I_band/self.ILS_Gaussian_term_sum[i])[::-1]

            #For analytic Jacobians
            if jacobians:
              Rc_band_albedo = (Sc_I_band_albedo/self.ILS_Gaussian_term_sum[i])[::-1]
              Rc_band_aerosol = (Sc_I_band_aerosol/self.ILS_Gaussian_term_sum[i])[::-1]
              Rc_band_q = (Sc_I_band_q/self.ILS_Gaussian_term_sum[i][:,None])[::-1,:]
              Rc_band_co2 = (Sc_I_band_co2/self.ILS_Gaussian_term_sum[i][:,None])[::-1,:]
              Rc_band_ch4 = (Sc_I_band_ch4/self.ILS_Gaussian_term_sum[i][:,None])[::-1,:]

            #Append for the band we're on
            self.R_band.append(Rc_band)
            if jacobians:
              #For analytic Jacobians
              self.R_band_albedo.append(Rc_band_albedo)
              self.R_band_aerosol.append(Rc_band_aerosol)
              self.R_band_q.append(Rc_band_q)
              self.R_band_co2.append(Rc_band_co2)
              self.R_band_ch4.append(Rc_band_ch4)

        self.y = np.concatenate(self.R_band)
        if jacobians:
          self.y_albedo = np.concatenate(self.R_band_albedo)
          self.y_aerosol = np.concatenate(self.R_band_aerosol)
          self.y_q = np.concatenate(self.R_band_q)
          self.y_co2 = np.concatenate(self.R_band_co2)
          self.y_ch4 = np.concatenate(self.R_band_ch4)

        noise = []
        for i in range(len(self.band_max_wn)):
          signal = self.R_band[i].max()
          sigma = signal/self.SNR
          np.random.seed(0)
          noise_temp = np.random.normal(0,sigma,self.band_spectral_points[i])
          noise.append(noise_temp)

          #If we're adding noise:
          if self.measurement_error: self.R_band[i] = self.R_band[i] + noise_temp

        #Now combine into one spectra
        self.y = np.concatenate(self.R_band)

        #Calculate Sy even if we didn't add noise because we need something for Sy
        noise_std = []
        for i in range(len(self.band_max_wn)):
          noise_std.append(np.full((len(noise[i])),np.std(noise[i])**2.0))
        Sy = np.diag(np.concatenate(noise_std))

        #Calculate this ahead of time so we don't have to calculate it every retrieval iteration
        Sy_inv = np.zeros(Sy.shape)
        np.fill_diagonal(Sy_inv,1./Sy.diagonal())

        self.Sy_inv = Sy_inv


    #Calculate Rayleigh scatter phase function expansion coefficients
    #rho = depolarization factor
    def rayleigh_coefs(self,rho):

      a = (1. - rho) / (2. + rho);

      coefs = np.zeros((6,3))

      coefs[0,0] = 1.;
      coefs[1,0] = 0.;
      coefs[2,0] = 0.;
      coefs[3,0] = 0.;
      coefs[4,0] = 0.;
      coefs[5,0] = 0.;

      coefs[0,1] = 0.;
      coefs[1,1] = 0.;
      coefs[2,1] = 0.;
      coefs[3,1] = 3. * (1. - 2. * rho) / (2. + rho);
      coefs[4,1] = 0.;
      coefs[5,1] = 0.;

      coefs[0,2] = a;
      coefs[1,2] =         6.  * a;
      coefs[2,2] = 0.;
      coefs[3,2] = 0.;
      coefs[4,2] = np.sqrt(6.) * a;
      coefs[5,2] = 0.;

      return coefs


    #Return erosol scatter phase function expansion coefficients for i_band for:
    #    d_aerosol = 3.0e-6 #Particle diameter [m]
    #    n_aerosol = 1.4 + 0.0003j #Refractive index
    def aerosol_coefs(self,i_band):

      if i_band == 0:
          n_coefs = 28
          coefs   = np.reshape(np.array(
              [1.000000e+00, 0.000000e+00, 0.000000e+00, 8.965733e-01,  0.000000e+00,  0.000000e+00,
               1.883236e+00, 0.000000e+00, 0.000000e+00, 1.971691e+00,  0.000000e+00,  0.000000e+00,
               2.448646e+00, 3.717850e+00, 3.460467e+00, 2.380890e+00, -1.094226e-01,  1.090392e-01,
               2.066593e+00, 2.674984e+00, 2.679724e+00, 2.136218e+00, -7.975887e-02, -1.263843e-01,
               2.148881e+00, 2.499134e+00, 2.370181e+00, 2.083182e+00, -1.958896e-01,  1.507534e-01,
               1.977387e+00, 2.058566e+00, 2.056800e+00, 1.978192e+00, -1.484463e-01, -1.618268e-01,
               2.028490e+00, 2.235331e+00, 2.255672e+00, 2.097312e+00, -5.919315e-03,  1.959873e-01,
               2.337513e+00, 2.214607e+00, 2.072087e+00, 2.214199e+00, -1.364870e-01, -1.938318e-01,
               2.374262e+00, 2.576643e+00, 2.703696e+00, 2.526166e+00,  8.585937e-02,  2.216301e-01,
               2.957218e+00, 2.751720e+00, 2.586271e+00, 2.829103e+00, -1.142909e-01, -2.037027e-01,
               2.971642e+00, 3.225419e+00, 3.297431e+00, 3.055808e+00,  9.780695e-02,  1.577388e-01,
               3.552607e+00, 3.295906e+00, 3.257302e+00, 3.555216e+00, -1.032539e-01, -1.771686e-01,
               3.517939e+00, 3.851513e+00, 3.796663e+00, 3.489108e+00, -4.428352e-03, -3.188984e-02,
               3.851999e+00, 3.589441e+00, 3.620874e+00, 3.913339e+00, -5.225722e-02, -1.257314e-01,
               3.269958e+00, 3.728155e+00, 3.761797e+00, 3.368125e+00, -3.643060e-01, -4.927155e-01,
               2.379509e+00, 2.351352e+00, 2.341388e+00, 2.463459e+00,  9.093700e-02, -7.861866e-01,
               1.195550e+00, 1.603503e+00, 1.325530e+00, 9.593723e-01,  2.976680e-01, -7.426527e-01,
               5.592347e-02, 3.738040e-02, 3.063643e-02, 1.184672e-01,  3.125284e-01,  2.031676e-01,
               2.784969e-01, 3.191834e-01, 2.968767e-01, 2.614096e-01, -2.190491e-02, -3.068446e-02,
               6.793373e-02, 7.784103e-02, 7.591266e-02, 6.665464e-02,  1.338471e-02, -1.396770e-02,
               1.492211e-02, 1.714940e-02, 1.647092e-02, 1.445522e-02,  5.355730e-03, -2.993855e-03,
               2.845843e-03, 3.276105e-03, 3.041393e-03, 2.673972e-03,  1.375821e-03, -4.835834e-04,
               4.781650e-04, 5.506540e-04, 4.865804e-04, 4.292616e-04,  2.796379e-04, -6.321043e-05,
               7.202072e-05, 8.288328e-05, 6.879923e-05, 6.096317e-05,  4.811786e-05, -6.893304e-06,
               9.864289e-06, 1.133581e-05, 8.750373e-06, 7.791905e-06,  7.262595e-06, -6.395809e-07,
               1.241234e-06, 1.423563e-06, 1.015303e-06, 9.086562e-07,  9.824561e-07, -5.125031e-08,
               1.444330e-07, 1.652543e-07, 1.085751e-07, 9.764776e-08,  1.207423e-07, -3.591385e-09,
               1.560380e-08, 1.780545e-08, 1.077637e-08, 9.736841e-09,  1.359733e-08, -2.225526e-10]
          ), (6, n_coefs), order='F')

      elif i_band == 1:
          n_coefs = 17
          coefs   = np.reshape(np.array(
              [1.000000e+00, 0.000000e+00, 0.000000e+00, 9.558799e-01,  0.000000e+00,  0.000000e+00,
               2.409701e+00, 0.000000e+00, 0.000000e+00, 2.418855e+00,  0.000000e+00,  0.000000e+00,
               3.169379e+00, 4.210609e+00, 4.125717e+00, 3.146999e+00, -2.834060e-02, -2.633304e-03,
               3.280668e+00, 3.930733e+00, 3.907318e+00, 3.290329e+00, -1.022581e-01, -1.943394e-01,
               2.879395e+00, 3.561741e+00, 3.532469e+00, 2.913491e+00, -1.126528e-01, -1.841096e-01,
               2.195789e+00, 2.663273e+00, 2.625718e+00, 2.250362e+00, -1.473459e-01, -4.165110e-01,
               1.259739e+00, 1.746651e+00, 1.646359e+00, 1.226155e+00, -1.937594e-02, -4.485695e-01,
               4.776816e-01, 6.409883e-01, 6.246772e-01, 4.892794e-01,  1.875980e-01, -2.495802e-01,
               1.687875e-01, 2.354579e-01, 2.057944e-01, 1.573723e-01,  9.183152e-02, -5.066795e-02,
               4.093730e-02, 5.719828e-02, 4.623247e-02, 3.548128e-02,  2.931686e-02, -8.476874e-03,
               7.657404e-03, 1.060186e-02, 7.655445e-03, 5.957974e-03,  6.454270e-03, -9.872941e-04,
               1.147847e-03, 1.566952e-03, 9.928909e-04, 7.861141e-04,  1.066287e-03, -8.478276e-05,
               1.421760e-04, 1.910259e-04, 1.052879e-04, 8.482397e-05,  1.401194e-04, -5.610737e-06,
               1.488449e-05, 1.968033e-05, 9.417505e-06, 7.711422e-06,  1.521343e-05, -2.956389e-07,
               1.338088e-06, 1.742192e-06, 7.259227e-07, 6.031314e-07,  1.398857e-06, -1.273529e-08,
               1.044547e-07, 1.340538e-07, 4.891459e-08, 4.116337e-08,  1.107271e-07, -4.592677e-10,
               7.140902e-09, 9.043001e-09, 2.909752e-09, 2.476182e-09,  7.633099e-09, -1.420959e-11]
          ), (6, n_coefs), order='F')

      elif i_band == 2:
          n_coefs = 16
          coefs   = np.reshape(np.array(
              [1.000000e+00, 0.000000e+00, 0.000000e+00, 9.581467e-01,  0.000000e+00,  0.000000e+00,
               2.396714e+00, 0.000000e+00, 0.000000e+00, 2.422691e+00,  0.000000e+00,  0.000000e+00,
               3.145855e+00, 4.209885e+00, 4.110651e+00, 3.098868e+00, -3.414275e-02,  1.507560e-02,
               3.191527e+00, 3.850364e+00, 3.869172e+00, 3.230393e+00, -8.705805e-02, -1.816627e-01,
               2.764995e+00, 3.459737e+00, 3.409534e+00, 2.782140e+00, -1.388044e-01, -1.596200e-01,
               2.047142e+00, 2.504542e+00, 2.471884e+00, 2.084703e+00, -1.283519e-01, -4.046175e-01,
               1.121463e+00, 1.573477e+00, 1.498735e+00, 1.094324e+00, -1.662840e-03, -3.734151e-01,
               3.848706e-01, 5.318252e-01, 5.230739e-01, 3.954823e-01,  1.558268e-01, -1.966867e-01,
               1.191531e-01, 1.685719e-01, 1.465191e-01, 1.106957e-01,  7.250850e-02, -3.831371e-02,
               2.676896e-02, 3.777727e-02, 2.947041e-02, 2.251560e-02,  2.071487e-02, -5.521576e-03,
               4.670146e-03, 6.510265e-03, 4.462016e-03, 3.468096e-03,  4.146287e-03, -5.654324e-04,
               6.557419e-04, 8.993408e-04, 5.356333e-04, 4.241346e-04,  6.310243e-04, -4.312198e-05,
               7.626017e-05, 1.027899e-04, 5.298310e-05, 4.271387e-05,  7.701789e-05, -2.551338e-06,
               7.502862e-06, 9.942431e-06, 4.440000e-06, 3.638527e-06,  7.802962e-06, -1.209130e-07,
               6.338670e-07, 8.266012e-07, 3.212677e-07, 2.671185e-07,  6.710735e-07, -4.713455e-09,
               4.648053e-08, 5.971974e-08, 2.033172e-08, 1.712021e-08,  4.973655e-08, -1.548941e-10]
          ), (6, n_coefs), order='F')

      else:
          print('ERROR: Invalid band number: %d' % i_band, file=sys.stderr)
          exit()

      return n_coefs, coefs


    def aggregate_optical_props(self,
                                tau_gas_wl_layer,
                                tau_ray_wl_layer,
                                n_coefs_ray,
                                coefs_ray,
                                aerosol_layer_mask,
                                tau_aerosol_wl_layer,
                                n_coefs_aerosol,
                                ssa_aerosol,
                                coefs_aerosol):

      n_chans  = tau_gas_wl_layer.shape[0]
      n_layers = tau_gas_wl_layer.shape[1]

      tau_layer_wl = np.zeros((n_chans, n_layers))
      ssa_layer_wl = np.zeros((n_chans, n_layers))

      for j in range(n_layers):
          tau_layer_wl[:,j] = tau_gas_wl_layer[:,j] + \
                              tau_ray_wl_layer[:,j] + \
                              tau_aerosol_wl_layer[:,j]

      for j in range(n_layers):
          ssa_layer_wl[:,j] = (tau_gas_wl_layer[:,j] * 0. + \
                               tau_ray_wl_layer[:,j] * 1. + \
                               tau_aerosol_wl_layer[:,j] * ssa_aerosol[j]) / tau_layer_wl[:,j]

      n_gc_layer = np.zeros((n_chans, n_layers), dtype='int32')

      for j in range(n_layers):
          n_gc_layer[:,j] = n_coefs_ray
          if self.tau_aerosol and aerosol_layer_mask[j]:
              n_gc_layer[:,j] = np.maximum(n_gc_layer[:,j], n_coefs_aerosol)

      gc_layer = np.zeros((n_chans, n_layers, 6, np.amax(n_gc_layer)))

      for i in range(n_chans):
          for j in range(n_layers):
              gc_layer[i, j,:,0:n_coefs_ray] = tau_ray_wl_layer[i,j] * 1. * coefs_ray[:,:] / (tau_ray_wl_layer[i,j] * 1. + tau_aerosol_wl_layer[i,j] * ssa_aerosol[j])

              if self.tau_aerosol and aerosol_layer_mask[j]:
                  gc_layer[i, j,:,0:n_gc_layer[i,j]] += tau_aerosol_wl_layer[i,j] * ssa_aerosol[j] * coefs_aerosol[:,:] / (tau_ray_wl_layer[i,j] * 1. + tau_aerosol_wl_layer[i,j] * ssa_aerosol[j])

      return tau_layer_wl, ssa_layer_wl, n_gc_layer, gc_layer


    #Calculate intensities for a single band
    def intensity_xrtm(self,
                       i_band,
                       tau_gas_wl_layer,
                       tau_ray_wl_layer,
                       aerosol_layer_mask,
                       tau_aerosol,
                       ssa_aerosol,
                       P_aerosol,
                       qext_aerosol_band_0,
                       qext_aerosol,
                       albedo,
                       band_solar_irradiance):

      n_chans  = tau_gas_wl_layer.shape[0]
      n_layers = tau_gas_wl_layer.shape[1]


      #Dealing with divide by zero issues
      if qext_aerosol_band_0[0] == 0:
          qext_scaling = np.zeros((len(qext_aerosol)))
      else:
          qext_scaling = qext_aerosol/qext_aerosol_band_0[0]


      n_coefs_ray = 3
      coefs_ray   = self.rayleigh_coefs(rho_rayleigh)

      n_coefs_aerosol, \
      coefs_aerosol = self.aerosol_coefs(i_band)

      tau_aerosol_wl_layer = np.zeros((n_chans, n_layers))

      if self.tau_aerosol:
          tau_aerosol_d_n_lay = tau_aerosol * qext_scaling / aerosol_layer_mask.sum()

          for j in range(n_layers):
              if aerosol_layer_mask[j]:
                  tau_aerosol_wl_layer[:,j] += tau_aerosol_d_n_lay


      tau_layer_wl, ssa_layer_wl, n_gc_layer, gc_layer = \
          self.aggregate_optical_props(tau_gas_wl_layer,
                                       tau_ray_wl_layer,
                                       n_coefs_ray,
                                       coefs_ray,
                                       aerosol_layer_mask,
                                       tau_aerosol_wl_layer,
                                       n_coefs_aerosol,
                                       ssa_aerosol,
                                       coefs_aerosol)


      I         = np.zeros((n_chans))          #wn
      I_albedo  = np.zeros((n_chans))          #wn
      I_aerosol = np.zeros((n_chans))          #wn
      I_q       = np.zeros((n_chans,n_layers)) #wn x layers
      I_co2     = np.zeros((n_chans,n_layers)) #wn x layers
      I_ch4     = np.zeros((n_chans,n_layers)) #wn x layers

      for i in range(n_chans):
        I_p, I_m, K_p, K_m = self.call_xrtm_radiance(i,tau_layer_wl[i],ssa_layer_wl[i],n_gc_layer[i],gc_layer[0],albedo)

        I[i]        = I_p[0,0,0,0]
        I_albedo[i] = K_p[0,0,0,0,0]

        I[i]        = I[i]        * band_solar_irradiance
        I_albedo[i] = I_albedo[i] * band_solar_irradiance


      return I, I_albedo, I_aerosol, I_q, I_co2, I_ch4


    #Calculate intensities for a single band using xrtm
    def call_xrtm_radiance(self,
                           i_point,
                           tau_layer,
                           ssa_layer,
                           n_gc_layer,
                           gc_layer,
                           albedo):

        #*******************************************************************************
        # Define inputs.
        #*******************************************************************************
        options       = ['calc_derivs', 'delta_m', 'n_t_tms', 'output_at_levels', 'sfi', 'source_solar']

        solvers       = ['two_stream']

        max_coef      = np.amax(n_gc_layer)
        n_quad        = 1
        n_stokes      = 1
        n_derivs      = 1
        n_layers      = tau_layer.shape[0]
        n_theta_0s    = 1
        n_kernel_quad = 16
        kernels       = ['lambertian']
        n_out_levels  = 1
        n_out_thetas  = 1
        n_out_phis    = 1

        F_0           = 1.
        theta_0       = self.sza_0

        out_levels    = [0]

        out_thetas    = [self.sza]
        phi           = [self.phi]
        out_phis      = [phi]

        ltau          = tau_layer

        omega         = ssa_layer

        n_coef        = n_gc_layer
        coef          =   gc_layer

        albedo_       = albedo


        #*******************************************************************************
        # Create an XRTM instance.
        #*******************************************************************************
        try:
            model = xrtm.xrtm(options, solvers, max_coef, n_quad, n_stokes, n_derivs,
                    n_layers, n_theta_0s, n_kernel_quad, kernels, n_out_levels, n_out_thetas)
        except xrtm.error as e:
            print(str(e) + '\nERROR: xrtm.init()')
            exit()


        #*******************************************************************************
        # Set inputs.
        #
        # Inputs must be set before the first model run.  For subsequent runs only the
        # inputs that change need to be set.  For example calculating the radiance
        # across the O2-A band spectrum, assuming constant scattering properites, would
        # require only updating ltau and omega for each point.
        #*******************************************************************************
        try:
            model.set_fourier_tol(.0001)
        except xrtm.error as e:
            print(str(e) + '\nERROR: xrtm.set_fourier_tol()')
            exit()

        try:
            model.set_out_levels(out_levels)
        except xrtm.error as e:
            print(str(e) + '\nERROR: xrtm.set_out_levels()')
            exit()

        try:
            model.set_out_thetas(out_thetas)
        except xrtm.error as e:
            print(str(e) + '\nERROR: xrtm.set_out_thetas()')
            exit()

        try:
            model.set_F_iso_top(0.)
        except xrtm.error as e:
            print(str(e) + '\nERROR: xrtm.set_F_iso_top()')
            exit()

        try:
            model.set_F_iso_bot(0.)
        except xrtm.error as e:
            print(str(e) + '\nERROR: xrtm.set_F_iso_bot()')
            exit()

        try:
            model.set_F_0(F_0)
        except xrtm.error as e:
            print(str(e) + '\nERROR: xrtm.set_F_0()')
            exit()

        try:
            model.set_theta_0(theta_0)
        except xrtm.error as e:
            print(str(e) + '\nERROR: xrtm.set_theta_0()')
            exit()

        try:
            model.set_phi_0(0.)
        except xrtm.error as e:
            print(str(e) + '\nERROR: xrtm.set_phi_0()')
            exit()


        # Set optical property inputs
        try:
            model.set_ltau_n(ltau)
        except xrtm.error as e:
            print(str(e) + '\nERROR: xrtm.set_ltau_n()')
            exit()

        try:
            model.set_omega_n(omega)
        except xrtm.error as e:
            print(str(e) + '\nERROR: xrtm.set_omega_n()')
            exit()

        try:
            model.set_coef_n(n_coef, coef[:,0:1,:])
        except xrtm.error as e:
            print(str(e) + '\nERROR: xrtm.set_coef_n()')
            exit()

        # Set surface albedo
        try:
            model.set_kernel_ampfac(0, albedo_)
        except xrtm.error as e:
            print(str(e) + '\nERROR: xrtm.set_kernel_ampfac()')
            exit()


        #*******************************************************************************
        # Set linearized inputs.
        #*******************************************************************************
        '''
        try:
            model.set_ltau_l_11(2, 0, 1.)
        except xrtm.error as e:
            print(str(e) + '\nERROR: xrtm.set_ltau_l_11()')
            exit()

        try:
            model.set_omega_l_11(2, 1, 1.)
        except xrtm.error as e:
            print(str(e) + '\nERROR: xrtm.set_omega_l_11()')
            exit()
        '''
        try:
            model.set_kernel_ampfac_l_1(0, 0, 1.)
        except xrtm.error as e:
            print(str(e) + '\nERROR: xrtm.set_kernel_ampfac_l_1()')
            exit()

        try:
            model.update_varied_layers()
        except xrtm.error as e:
            print(str(e) + '\nERROR: model.update_varied_layers()')
            exit()


        #*******************************************************************************
        # Run the model for radiances and associated derivatives.  If this is the
        # initial run and all the required inputs have not been initialized then XRTM
        # will print(a appropriate message and return < 0.
        #*******************************************************************************
        try:
            I_p, I_m, K_p, K_m = model.radiance(solvers[0], n_out_phis, out_phis)
        except xrtm.error as e:
            print(str(e) + '\nERROR: model.radiance()')
            exit()


        #*******************************************************************************
        # Output results.
        #*******************************************************************************
        if i_point == 0:
            for i in range(0, n_out_levels):
                print('level: %d' % i)
                print('     intensity:')

                for j in range(0, n_out_thetas):
                    print('          theta = %9.2E, I_p = %13.6E, I_m = %13.6E' % \
                          (out_thetas[j], I_p[i,j,0,0], I_m[i,j,0,0]))
                for j in range(0, n_derivs):
                    print('     derivative: %d' % j)
                    for k in range(0, n_out_thetas):
                        print('          theta = %9.2E, K_p = %13.6E, K_m = %13.6E' % \
                              (out_thetas[k], K_p[i,j,k,0,0], K_m[i,j,k,0,0]))
            print()


        #*******************************************************************************
        # Delete xrtm instance.
        #*******************************************************************************
        del model


        return I_p, I_m, K_p, K_m


    #Assume a Gaussian ILS
    def spectral_response_function(self,band_wn_index,band,sigma_band,ILS_Gaussian_term,I_band,I_band_albedo,I_band_aerosol,I_band_q,I_band_co2,I_band_ch4,jacobians):

        Sc_I_band = np.zeros((len(band_wn_index))) #wn instrument

        Sc_I_band_albedo = np.zeros((len(band_wn_index))) #wn instrument
        Sc_I_band_aerosol = np.zeros((len(band_wn_index))) #wn instrument
        Sc_I_band_q = np.zeros((len(band_wn_index),I_band_q.shape[1])) #wn instrument x layers
        Sc_I_band_co2 = np.zeros((len(band_wn_index),I_band_co2.shape[1])) #wn instrument x layers
        Sc_I_band_ch4 = np.zeros((len(band_wn_index),I_band_ch4.shape[1])) #wn instrument x layers

        round_term = round(sigma_band*ILS_width*100.0)

        for i in range(len(band_wn_index)):

            #Dealing with the starting edge of the band
            if band[band_wn_index[i]] <= band[0]+sigma_band*ILS_width:

                j_index_temp_lower = 0
                j_index_temp_upper = int(band_wn_index[i]+round_term)

            #Dealing with the trailing edge of the band
            elif band[band_wn_index[i]] >= band[len(band)-1]-sigma_band*ILS_width:

                j_index_temp_lower = int(band_wn_index[i]-round_term)
                j_index_temp_upper = len(band)

            #Most of the band
            else:
                j_index_temp_lower = int(band_wn_index[i]-round_term)
                j_index_temp_upper = int(band_wn_index[i]+round_term)

            Sc_I_band[i] = np.sum(I_band[j_index_temp_lower:j_index_temp_upper] * ILS_Gaussian_term[i,j_index_temp_lower:j_index_temp_upper])

            if jacobians:
                Sc_I_band_albedo[i] = np.sum(I_band_albedo[j_index_temp_lower:j_index_temp_upper] * ILS_Gaussian_term[i,j_index_temp_lower:j_index_temp_upper])
                Sc_I_band_aerosol[i] = np.sum(I_band_aerosol[j_index_temp_lower:j_index_temp_upper] * ILS_Gaussian_term[i,j_index_temp_lower:j_index_temp_upper])
                Sc_I_band_q[i,:] = np.sum(I_band_q[j_index_temp_lower:j_index_temp_upper,:] * ILS_Gaussian_term[i,j_index_temp_lower:j_index_temp_upper,None],axis=0)
                Sc_I_band_co2[i,:] = np.sum(I_band_co2[j_index_temp_lower:j_index_temp_upper,:] * ILS_Gaussian_term[i,j_index_temp_lower:j_index_temp_upper,None],axis=0)
                Sc_I_band_ch4[i,:] = np.sum(I_band_ch4[j_index_temp_lower:j_index_temp_upper,:] * ILS_Gaussian_term[i,j_index_temp_lower:j_index_temp_upper,None],axis=0)

        return Sc_I_band, Sc_I_band_albedo, Sc_I_band_aerosol, Sc_I_band_q, Sc_I_band_co2, Sc_I_band_ch4


class Retrieval:
    '''Retrieval object. Parameter defaults are set in settings.py
    band_max_wn               [array] sfddsfds
    band_min_wn               [array] sdfdsf
    band_spectral_resolution  [array] fdsf

    '''

    def __init__(self):

        self.iterations = 0
        self.chisq_reduced_previous = 9999999.0

    def run(self, x, model_prior, model_true, absco_data, chisq_threshold=s.chisq_threshold):

        time_total=time.time()

        #Set the initial guess to the prior
        x["ret"] = deepcopy(x["prior"])

        self.chisq_threshold = chisq_threshold
        done=False
        while not done:

            #Time the iterations
            time1=time.time()

            if self.iterations == 0:
                print("-----------------------------------------------------------------------------------------------------------------------")
                print("State vector:")
                print("Name                                              Prior  True")
                for i in range(len(x["prior"])):
                    print(x["names"][i].ljust(49)+" "+str(x["prior"][i]).ljust(5)+"  "+str(x["true"][i]).ljust(20))
                print("-----------")

            #Calculate y using the state vector, x
            model_ret = self.forward_model(x, model_prior, absco_data, jacobians=True)

            #Calculate chisq
            chisq = (((model_true.y-model_ret.y).dot(model_true.Sy_inv)).dot((model_true.y-model_ret.y).T))+\
                (((x["ret"]-x["prior"]).dot(LA.inv(x["S_prior"]))).dot((x["ret"]-x["prior"]).T))
            self.chisq_reduced = chisq/len(model_true.y)

            #Check if our chisq_reduced is good enough to stop
            print("Reduced chisq = ",self.chisq_reduced)
            if self.chisq_reduced < self.chisq_threshold:
                print("Reduced chisq is less than "+str(self.chisq_threshold)+", so we're done!")
                done=True
                continue

            #Also stop if chisq is barely changing
            elif (self.chisq_reduced_previous - self.chisq_reduced) < 0.0001:
                print("Reduced chisq has stopped changing or increased, so we're done!")
                done=True
                continue

            else: print("Reduced chisq is not small enough yet, so update the state vector")
            self.chisq_reduced_previous = self.chisq_reduced

            #If not, loop through the state vector to calculate the Jacobian (K)
            self.K = np.zeros((len(model_true.y),len(x["ret"])))
            for i in range(len(x["ret"])):

                #Need to use finite differencing for T and p Jacobians
                if ("Temperature" in x["names"][i]) or ("Pressure" in x["names"][i]):
                    x_perturbed = deepcopy(x)
                    x_perturbed["ret"][i] += s.perturbation
                    model_perturbed = self.forward_model(x_perturbed, model_prior, absco_data, jacobians=False)
                    self.K[:,i] = ((model_perturbed.y - model_ret.y)/s.perturbation)

                #If we have the analytical derivative for the current state, use it!
                else: self.K[:,i] = model_ret.y_k[:,i]

            #Calculate the current error covariance matrix (S)
            self.S = LA.inv((self.K.T).dot(model_true.Sy_inv).dot(self.K) + LA.inv(x["S_prior"]))

            #If we haven't converged, determine how we want to perturb the state vector to try again
            x["dx"] = self.S.dot((((self.K.T).dot(model_true.Sy_inv).dot(model_true.y - model_ret.y)) - (LA.inv(x["S_prior"]).dot(x["ret"]-x["prior"]))))

            print("Updated state vector (iteration "+str(self.iterations+1)+"):")
            print("Name                                             Prior  True                  Current               Dx from prev. iteration:")
            for i in range(len(x["ret"])):
                print(x["names"][i].ljust(49)+str(x["prior"][i]).ljust(5)+"  "+str(x["true"][i]).ljust(20)+"  "+str(x["ret"][i] + x["dx"][i]).ljust(20)+"  "+str(x["dx"][i]))

            #Modify the state vector
            x["ret"] += x["dx"]

            #Print things. Index of CO2/CH4 depend on the forward model used.
            if "CO2 Profile Scale Factor" in x["names"] and "CH4 Profile Scale Factor" in x["names"]:
                print("-----------")
                print("Prior XCO2 =".ljust(31),'{:.5f}'.format(model_prior.xco2).rjust(10),"ppm")
                print("True XCO2 =".ljust(31),'{:.5f}'.format(model_true.xco2).rjust(10),"ppm")
                print("Current retrieved XCO2 =".ljust(31),'{:.5f}'.format(calculate_Xgas(model_prior.co2*x["ret"][0], model_prior.p*x["ret"][3], model_prior.q*x["ret"][4])[0] * 1e6).rjust(10),"ppm")
                print("XCO2 error (retrieved - true) =".ljust(31),'{:.5f}'.format(calculate_Xgas(model_prior.co2*x["ret"][0], model_prior.p*x["ret"][3], model_prior.q*x["ret"][4])[0] * 1e6 - model_true.xco2).rjust(10),"ppm")

                print("-----------")
                print("Prior XCH4 =".ljust(31),'{:.5f}'.format(model_prior.xch4).rjust(10),"ppb")
                print("True XCH4 =".ljust(31),'{:.5f}'.format(model_true.xch4).rjust(10),"ppb")
                print("Current retrieved XCH4 =".ljust(31),'{:.5f}'.format(calculate_Xgas(model_prior.ch4*x["ret"][1], model_prior.p*x["ret"][3], model_prior.q*x["ret"][4])[0] * 1e9).rjust(10),"ppb")
                print("XCH4 error (retrieved - true) =".ljust(31),'{:.5f}'.format(calculate_Xgas(model_prior.ch4*x["ret"][1], model_prior.p*x["ret"][3], model_prior.q*x["ret"][4])[0] * 1e9 - model_true.xch4).rjust(10),"ppb")
                print("-----------")

            elif "CO2 Profile Scale Factor" in x["names"] and "CH4 Profile Scale Factor" not in x["names"]:
                print("-----------")
                print("Prior XCO2 =".ljust(31),'{:.5f}'.format(model_prior.xco2).rjust(10),"ppm")
                print("True XCO2 =".ljust(31),'{:.5f}'.format(model_true.xco2).rjust(10),"ppm")
                print("Current retrieved XCO2 =".ljust(31),'{:.5f}'.format(calculate_Xgas(model_prior.co2*x["ret"][0], model_prior.p*x["ret"][2], model_prior.q*x["ret"][3])[0] * 1e6).rjust(10),"ppm")
                print("XCO2 error (retrieved - true) =".ljust(31),'{:.5f}'.format(calculate_Xgas(model_prior.co2*x["ret"][0], model_prior.p*x["ret"][2], model_prior.q*x["ret"][3])[0] * 1e6 - model_true.xco2).rjust(10),"ppm")
                print("-----------")

            elif "CO2 Profile Scale Factor" not in x["names"] and "CH4 Profile Scale Factor" in x["names"]:
                print("-----------")
                print("Prior XCH4 =".ljust(31),'{:.5f}'.format(model_prior.xch4).rjust(10),"ppb")
                print("True XCH4 =".ljust(31),'{:.5f}'.format(model_true.xch4).rjust(10),"ppb")
                print("Current retrieved XCH4 =".ljust(31),'{:.5f}'.format(calculate_Xgas(model_prior.ch4*x["ret"][0], model_prior.p*x["ret"][2], model_prior.q*x["ret"][3])[0] * 1e9).rjust(10),"ppb")
                print("XCH4 error (retrieved - true) =".ljust(31),'{:.5f}'.format(calculate_Xgas(model_prior.ch4*x["ret"][0], model_prior.p*x["ret"][2], model_prior.q*x["ret"][3])[0] * 1e9 - model_true.xch4).rjust(10),"ppb")
                print("-----------")

            else:
                print("Unexpected state vector setup!")

            print("Time for iteration",self.iterations+1,"=",'{:.2f}'.format(time.time()-time1), "s")
            print("-----------------------------------------------------------------------------------------------------------------------")

            #Add iteration
            self.iterations += 1

            if self.iterations > 5:
                print("More than 5 iterations, so we're stopping...")
                done=True
                continue

        print("-----------------------------------------------------------------------------------------------------------------------")
        print("Final reduced chisq = ",self.chisq_reduced)
        if "CO2 Profile Scale Factor" in x["names"] and "CH4 Profile Scale Factor" in x["names"]:
            xco2_ret_temp = calculate_Xgas(model_prior.co2*x["ret"][0], model_prior.p*x["ret"][3], model_prior.q*x["ret"][4])[0] * 1e6
            xch4_ret_temp = calculate_Xgas(model_prior.ch4*x["ret"][1], model_prior.p*x["ret"][3], model_prior.q*x["ret"][4])[0] * 1e9
            print("Final retrieved XCO2 =".ljust(23),'{:.5f}'.format(xco2_ret_temp).rjust(10),"+/-",'{:.5f}'.format(xco2_ret_temp*np.diagonal(self.S)[0]**0.5),"ppm")
            print("Final XCO2 error (retrieved - true) =".ljust(37),'{:.5f}'.format(xco2_ret_temp - model_true.xco2).rjust(8),"ppm")
            print("Final retrieved XCH4 =".ljust(23),'{:.5f}'.format(xch4_ret_temp).rjust(10),"+/-",'{:.5f}'.format(xch4_ret_temp*np.diagonal(self.S)[1]**0.5),"ppb")
            print("Final XCH4 error (retrieved - true) =".ljust(37),'{:.5f}'.format(xch4_ret_temp - model_true.xch4).rjust(8),"ppb")

        elif "CO2 Profile Scale Factor" in x["names"] and "CH4 Profile Scale Factor" not in x["names"]:
            xco2_ret_temp = calculate_Xgas(model_prior.co2*x["ret"][0], model_prior.p*x["ret"][2], model_prior.q*x["ret"][3])[0] * 1e6
            print("Final retrieved XCO2 =".ljust(23),'{:.5f}'.format(xco2_ret_temp).rjust(10),"+/-",'{:.5f}'.format(xco2_ret_temp*np.diagonal(self.S)[0]**0.5),"ppm")
            print("Final XCO2 error (retrieved - true) =".ljust(37),'{:.5f}'.format(xco2_ret_temp - model_true.xco2).rjust(8),"ppm")

        elif "CO2 Profile Scale Factor" not in x["names"] and "CH4 Profile Scale Factor" in x["names"]:
            xch4_ret_temp = calculate_Xgas(model_prior.ch4*x["ret"][0], model_prior.p*x["ret"][2], model_prior.q*x["ret"][3])[0] * 1e9
            print("Final retrieved XCH4 =".ljust(23),'{:.5f}'.format(xch4_ret_temp).rjust(10),"+/-",'{:.5f}'.format(xch4_ret_temp*np.diagonal(self.S)[0]**0.5),"ppb")
            print("Final XCH4 error (retrieved - true) =".ljust(37),'{:.5f}'.format(xch4_ret_temp - model_true.xch4).rjust(8),"ppb")

        else:
            print("Unexpected state vector setup!")

        print("Total retrieval time =",'{:.2f}'.format(time.time()-time_total),"s")


    def forward_model(self, x, model_prior, absco_data, jacobians=False):

        #Modify the prior state vector appropriately
        #Full-physics setup:
        if "CO2 Profile Scale Factor" in x["names"] and "CH4 Profile Scale Factor" in x["names"]:
            co2 = model_prior.co2 * x["ret"][0]
            ch4 = model_prior.ch4 * x["ret"][1]
            T = model_prior.T + x["ret"][2]
            p = model_prior.p * x["ret"][3]
            q = model_prior.q * x["ret"][4]
            albedo_band_1 = x["ret"][5]
            albedo_band_2 = x["ret"][6]
            albedo_band_3 = x["ret"][7]
            albedo = [albedo_band_1, albedo_band_2, albedo_band_3]

            #Set tau_aerosol appropriately
            if "Aerosol Optical Depth" in x["names"]: tau_aerosol = x["ret"][8]
            else: tau_aerosol = None

            #Call the foward function with info from the prior and the updated state vector elements
            model = ForwardFunction(SNR=model_prior.SNR,sza_0=model_prior.sza_0,sza=model_prior.sza,co2=co2,ch4=ch4,T=T,p=p,q=q,albedo=albedo,band_min_wn=model_prior.band_min_wn,band_max_wn=model_prior.band_max_wn,band_spectral_resolutions=model_prior.band_spectral_resolutions,band_min_um=model_prior.band_min_um,band_max_um=model_prior.band_max_um,band_spectral_points=model_prior.band_spectral_points,band_wn=model_prior.band_wn,band_wl=model_prior.band_wl,band_absco_res_wn=model_prior.band_absco_res_wn,resolving_power_band=model_prior.resolving_power_band,sigma_band=model_prior.sigma_band,band_wn_index=model_prior.band_wn_index,ILS_Gaussian_term=model_prior.ILS_Gaussian_term,ILS_Gaussian_term_sum=model_prior.ILS_Gaussian_term_sum,absco_data=absco_data,band_molecules=model_prior.band_molecules,P_aerosol=model_prior.P_aerosol,ssa_aerosol=model_prior.ssa_aerosol,qext_aerosol=model_prior.qext_aerosol,height_aerosol=model_prior.height_aerosol,tau_aerosol=tau_aerosol,jacobians=jacobians)

            #Calculate analytical derivative
            model.y_k = np.zeros((len(model.y),len(x["ret"])))

            if jacobians:
                model.y_k[:,0] = np.sum(model_prior.co2_layer[None,:] * model.y_co2, axis=1) #co2
                model.y_k[:,1] = np.sum(model_prior.ch4_layer[None,:] * model.y_ch4, axis=1) #ch4
                #model.y_k[:,2] #Calculate using finite differencing #T
                #model.y_k[:,3] #Calculate using finite differencing #p
                model.y_k[:,4] = np.sum(model_prior.q_layer[None,:] * model.y_q, axis=1) #q
                model.y_k[:,5] = np.concatenate((model.R_band_albedo[0],np.zeros((len(model.R_band_albedo[1]))),np.zeros((len(model.R_band_albedo[2]))))) #Band 1 albedo
                model.y_k[:,6] = np.concatenate((np.zeros((len(model.R_band_albedo[0]))),model.R_band_albedo[1],np.zeros((len(model.R_band_albedo[2]))))) #Band 2 albedo
                model.y_k[:,7] = np.concatenate((np.zeros((len(model.R_band_albedo[0]))),np.zeros((len(model.R_band_albedo[1]))),model.R_band_albedo[2])) #Band 3 albedo
                if "Aerosol Optical Depth" in x["names"]: model.y_k[:,8] = model.y_aerosol #tau_aerosol

        #CO2-only run
        elif "CO2 Profile Scale Factor" in x["names"] and "CH4 Profile Scale Factor" not in x["names"]:
            co2 = model_prior.co2 * x["ret"][0]
            T = model_prior.T + x["ret"][1]
            p = model_prior.p * x["ret"][2]
            q = model_prior.q * x["ret"][3]
            albedo_band_2 = x["ret"][4]
            albedo = [albedo_band_2]
            tau_aerosol = None

            #Call the foward function with info from the prior and the updated state vector elements
            model = ForwardFunction(SNR=model_prior.SNR,sza_0=model_prior.sza_0,sza=model_prior.sza,co2=co2,ch4=np.zeros(len(model_prior.ch4)),T=T,p=p,q=q,albedo=albedo,band_min_wn=model_prior.band_min_wn,band_max_wn=model_prior.band_max_wn,band_spectral_resolutions=model_prior.band_spectral_resolutions,band_min_um=model_prior.band_min_um,band_max_um=model_prior.band_max_um,band_spectral_points=model_prior.band_spectral_points,band_wn=model_prior.band_wn,band_wl=model_prior.band_wl,band_absco_res_wn=model_prior.band_absco_res_wn,resolving_power_band=model_prior.resolving_power_band,sigma_band=model_prior.sigma_band,band_wn_index=model_prior.band_wn_index,ILS_Gaussian_term=model_prior.ILS_Gaussian_term,ILS_Gaussian_term_sum=model_prior.ILS_Gaussian_term_sum,absco_data=absco_data,band_molecules=model_prior.band_molecules,P_aerosol=model_prior.P_aerosol,ssa_aerosol=model_prior.ssa_aerosol,qext_aerosol=model_prior.qext_aerosol,height_aerosol=model_prior.height_aerosol,tau_aerosol=tau_aerosol,jacobians=jacobians)

            #Calculate analytical derivative
            model.y_k = np.zeros((len(model.y),len(x["ret"])))

            if jacobians:
                model.y_k[:,0] = np.sum(model_prior.co2_layer[None,:] * model.y_co2, axis=1) #co2
                #model.y_k[:,1] #Calculate using finite differencing #T
                #model.y_k[:,2] #Calculate using finite differencing #p
                model.y_k[:,3] = np.sum(model_prior.q_layer[None,:] * model.y_q, axis=1) #q
                model.y_k[:,4] = model.R_band_albedo[0] #Band 2 albedo. 0th index in this case.

        #CH4-only run
        elif "CO2 Profile Scale Factor" not in x["names"] and "CH4 Profile Scale Factor" in x["names"]:
            ch4 = model_prior.ch4 * x["ret"][0]
            T = model_prior.T + x["ret"][1]
            p = model_prior.p * x["ret"][2]
            q = model_prior.q * x["ret"][3]
            albedo_band_3 = x["ret"][4]
            albedo = [albedo_band_3]
            tau_aerosol = None

            #Call the foward function with info from the prior and the updated state vector elements
            model = ForwardFunction(SNR=model_prior.SNR,sza_0=model_prior.sza_0,sza=model_prior.sza,co2=np.zeros(len(model_prior.co2)),ch4=ch4,T=T,p=p,q=q,albedo=albedo,band_min_wn=model_prior.band_min_wn,band_max_wn=model_prior.band_max_wn,band_spectral_resolutions=model_prior.band_spectral_resolutions,band_min_um=model_prior.band_min_um,band_max_um=model_prior.band_max_um,band_spectral_points=model_prior.band_spectral_points,band_wn=model_prior.band_wn,band_wl=model_prior.band_wl,band_absco_res_wn=model_prior.band_absco_res_wn,resolving_power_band=model_prior.resolving_power_band,sigma_band=model_prior.sigma_band,band_wn_index=model_prior.band_wn_index,ILS_Gaussian_term=model_prior.ILS_Gaussian_term,ILS_Gaussian_term_sum=model_prior.ILS_Gaussian_term_sum,absco_data=absco_data,band_molecules=model_prior.band_molecules,P_aerosol=model_prior.P_aerosol,ssa_aerosol=model_prior.ssa_aerosol,qext_aerosol=model_prior.qext_aerosol,height_aerosol=model_prior.height_aerosol,tau_aerosol=tau_aerosol,jacobians=jacobians)

            #Calculate analytical derivative
            model.y_k = np.zeros((len(model.y),len(x["ret"])))

            if jacobians:
                model.y_k[:,0] = np.sum(model_prior.ch4_layer[None,:] * model.y_ch4, axis=1) #ch4
                #model.y_k[:,1] #Calculate using finite differencing #T
                #model.y_k[:,2] #Calculate using finite differencing #p
                model.y_k[:,3] = np.sum(model_prior.q_layer[None,:] * model.y_q, axis=1) #q
                model.y_k[:,4] = model.R_band_albedo[0] #Band 3 albedo. 0th index in this case.

        else:
          print("Unexpected state vector setup!")

        return model


#Function to calculate column-mean dry-air mole-fraction of a gas. Assume gravity is constant with height.
def calculate_Xgas(gas_level,p_level,q_level):

    #Calculate the d(pressure) of a given layer
    p_diff=np.empty((len(p_level)-1))
    for i in range(len(p_level)-1):
        p_diff[i] = p_level[i+1]-p_level[i]

    gas_layer=np.empty((len(gas_level)-1)) #Layer gas concentrations
    q_layer=np.empty((len(q_level)-1)) #Layer specific humidities
    for i in range(len(q_level)-1):
        gas_layer[i] = (gas_level[i+1]+gas_level[i])/2.0
        q_layer[i] = (q_level[i+1]+q_level[i])/2.0

    #Calculate the column density of dry air per unit pressure
    c_layer = (1.-q_layer)/g/M

    #Calcualte the pressure weighting function (PWF)
    h = (c_layer * p_diff) / np.sum(c_layer * p_diff)

    #Calculate the column-average dry-air mole-fraction of the gas
    xgas = np.sum(h * gas_layer)

    return xgas, h


#Find the layer average for a given array of levels
def layer_average(levels):
    layers=np.empty((len(levels)-1)) #Layer pressures
    for i in range(len(levels)-1):
        layers[i] = (levels[i+1]+levels[i])/2.0
    return layers


#Calculate the solar flux for each band's average wavelength. Ignoring all other complexities with the solar spectrum for now.
def planck(T_temp,wl_temp):
    B_temp=2.0*h*c**2.0/wl_temp**5.0/(np.exp(h*c/k/wl_temp/T_temp)-1.0)
    return B_temp


#Calculate the number of dry air molecules
def calculate_n_dry_air(p_diff,q_layer):
  n_dry_air = p_diff * Na / M / g * (1.-q_layer)
  return n_dry_air


