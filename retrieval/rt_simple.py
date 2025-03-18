import numpy as np
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


################################################################################
# A simple two way extinction RTM with single scattering from an aerosol layer
# and the surface. 
################################################################################
class RTSimple:

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

#         print('<<<<<<<<<<<<<<<<<<<< simple rt')

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
    #         I_q  [i,:] = band_solar_irradiances/np.pi * (albedo*mu_0*exp_term[i] * (-m) * tau_star_band_q[i,:] + ssa_aerosol[i]*P_aerosol[i]*tau_aerosol[i]*qext_scaling[i]*exp_term_above_aerosol[i]/4./mu * (-m) * tau_above_aerosol_star_band_q[i,:])

              I_q  [i,:] = band_solar_irradiances/np.pi * (-m*(albedo*mu_0*exp_term[i] * tau_star_band_q  [i,:] + ssa_aerosol[i]*P_aerosol[i]*tau_aerosol[i]*qext_scaling[i]*exp_term_above_aerosol[i]/4./mu * tau_above_aerosol_star_band_q  [i,:]))
              I_co2[i,:] = band_solar_irradiances/np.pi * (-m*(albedo*mu_0*exp_term[i] * tau_star_band_co2[i,:] + ssa_aerosol[i]*P_aerosol[i]*tau_aerosol[i]*qext_scaling[i]*exp_term_above_aerosol[i]/4./mu * tau_above_aerosol_star_band_co2[i,:]))
              I_ch4[i,:] = band_solar_irradiances/np.pi * (-m*(albedo*mu_0*exp_term[i] * tau_star_band_ch4[i,:] + ssa_aerosol[i]*P_aerosol[i]*tau_aerosol[i]*qext_scaling[i]*exp_term_above_aerosol[i]/4./mu * tau_above_aerosol_star_band_ch4[i,:]))

          return I, I_albedo, I_aerosol, I_q, I_co2, I_ch4
