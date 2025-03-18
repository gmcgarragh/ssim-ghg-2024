import numpy as np
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import xrtm


# Constants for Rayleigh scattering calculations
rho_rayleigh = 0.0279 # depolarization factor


aggregate_gc        = True
aggregate_gc_derivs = False


################################################################################
# XRTM with a full multiple scattering solution and optionally polarization.
#
# See https://reef.atmos.colostate.edu/~gregm/xrtm/ for code and documentation.
################################################################################
class RTXrtm:

    def __init__(self, sza_0, sza, phi):

        self.sza_0 = sza_0
        self.sza = sza
        self.phi = phi


    ############################################################################
    # Calculate Rayleigh scatter phase function expansion coefficients
    #     rho = depolarization factor
    ############################################################################
    def rayleigh_coefs(self, rho):

        a = (1. - rho) / (2. + rho)

        coefs = np.zeros((6,3))

        coefs[0,0] = 1.
        coefs[1,0] = 0.
        coefs[2,0] = 0.
        coefs[3,0] = 0.
        coefs[4,0] = 0.
        coefs[5,0] = 0.

        coefs[0,1] = 0.
        coefs[1,1] = 0.
        coefs[2,1] = 0.
        coefs[3,1] = 3. * (1. - 2. * rho) / (2. + rho)
        coefs[4,1] = 0.
        coefs[5,1] = 0.

        coefs[0,2] = a
        coefs[1,2] =         6.  * a
        coefs[2,2] = 0.
        coefs[3,2] = 0.
        coefs[4,2] = np.sqrt(6.) * a
        coefs[5,2] = 0.

        return coefs


    ############################################################################
    # Return aerosol scattering phase function expansion coefficients for i_band
    # for:
    #     d_aerosol = 3.0e-6        # Particle diameter [m]
    #     n_aerosol = 1.4 + 0.0003j # Refractive index
    #
    # Note: This only applies to the above aerosol parameters as demonstration.
    # For other aerosol types and/or sizes the scattering parameters can be
    # obtained from an external source, lookup table for example, or calculated
    # on-the-fly with a Mie code capable of computing the phase matrix expansion
    # coefficients in terms of generalized spherical functions.
    ############################################################################
    def aerosol_coefs(self, i_band):

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


    ############################################################################
    # Aggregate optical thickness, single scattering albedo, and phase matrix
    # expansion coefficients for gas absorption, Rayleigh scattering, and
    # particle absorption and scattering.
    ############################################################################
    def aggregate_optical_props(self,
                                tau_gas_wl_layer,
                                tau_gas_wl_layer_l,
                                tau_ray_wl_layer,
                                tau_ray_wl_layer_l,
                                n_elem,
                                n_coefs_ray,
                                coefs_ray,
                                coefs_ray_l,
                                aer_layer_mask,
                                tau_aer_wl_layer,
                                tau_aer_wl_layer_l,
                                ssa_aer_wl_layer,
                                ssa_aer_wl_layer_l,
                                n_coefs_aer,
                                coefs_aer,
                                coefs_aer_l):

        n_chans  = tau_gas_wl_layer.shape[0]
        n_layers = tau_gas_wl_layer.shape[1]
        n_derivs = tau_gas_wl_layer_l.shape[2]

        tau_layer_wl   = np.zeros((n_chans, n_layers))
        ssa_layer_wl   = np.zeros((n_chans, n_layers))

        tau_layer_wl_l = np.zeros((n_chans, n_layers, n_derivs))
        ssa_layer_wl_l = np.zeros((n_chans, n_layers, n_derivs))


        # Aggregate optical thicknesses and associated derivatives

        for j in range(n_layers):
            tau_layer_wl[:,j] = tau_gas_wl_layer[:,j] + \
                                tau_ray_wl_layer[:,j] + \
                                tau_aer_wl_layer[:,j]
            for k in range(n_derivs):
                tau_layer_wl_l[:,j,k] = tau_gas_wl_layer_l[:,j,k] + \
                                        tau_ray_wl_layer_l[:,j,k] + \
                                        tau_aer_wl_layer_l[:,j,k]


        # Aggregate single scattering albedos and associated derivatives

        for j in range(n_layers):
            a = tau_ray_wl_layer[:,j] * 1. + tau_aer_wl_layer[:,j] * ssa_aer_wl_layer[:,j]

            ssa_layer_wl[:,j] = a / tau_layer_wl[:,j]

            for k in range(n_derivs):
                ssa_layer_wl_l[:,j,k] = (tau_ray_wl_layer_l[:,j,k] * 1. + \
                                         tau_aer_wl_layer_l[:,j,k] * ssa_aer_wl_layer[:,j] + tau_aer_wl_layer[:,j] * ssa_aer_wl_layer_l[:,j,k]) / tau_layer_wl[:,j] - \
                                         a * tau_layer_wl_l[:,j,k] / (tau_layer_wl[:,j] * tau_layer_wl[:,j])


        # Aggregate phase matrix expansion coefficients and associated derivatives

        n_gc_layer = np.zeros((n_chans, n_layers), dtype='int32')

        for j in range(n_layers):
            n_gc_layer[:,j] = n_coefs_ray

            if aer_layer_mask[j]:
                n_gc_layer[:,j] = np.maximum(n_gc_layer[:,j], n_coefs_aer)

        gc_layer   = np.zeros((n_chans, n_layers,           n_elem, np.amax(n_gc_layer)))
        gc_layer_l = np.zeros((n_chans, n_layers, n_derivs, n_elem, np.amax(n_gc_layer)))

        for i in range(n_chans):
            for j in range(n_layers):
#               print(i,j)

                gc_layer[i,j,:,0] = 1.

                if aggregate_gc:
                    a = tau_ray_wl_layer[i,j] * 1. * coefs_ray[0:n_elem,:]

                    b = tau_ray_wl_layer[i,j] * 1. + tau_aer_wl_layer[i,j] * ssa_aer_wl_layer[i,j]
                    c = b * b

                    d = tau_ray_wl_layer_l[i,j,:] * 1. + tau_aer_wl_layer_l[i,j,:] * ssa_aer_wl_layer[i,j] + tau_aer_wl_layer[i,j] * ssa_aer_wl_layer_l[i,j,:]

                    gc_layer[i,j,:,0:n_coefs_ray] = a / b

                    if aggregate_gc_derivs:
                        for k in range(n_derivs):
                            gc_layer_l[i,j,k,:,0:n_coefs_ray] = (tau_ray_wl_layer_l[i,j,k] * 1. * coefs_ray    [0:n_elem,:] +
                                                                 tau_ray_wl_layer  [i,j  ] * 1. * coefs_ray_l[k,0:n_elem,:]) / b - \
                                                                 \
                                                                 a * \
                                                                 d[k] / c
                            '''
                            gc_layer_l[i,j,k,:,0:n_coefs_ray] = (tau_ray_wl_layer_l[i,j,k] * 1. * coefs_ray  [i,j,0:n_elem,:] +
                                                                 tau_ray_wl_layer  [i,j  ] * 1. * coefs_ray_l[i,j,k,0:n_elem,:]) / b - \
                                                                 \
                                                                 a * \
                                                                 d[k] / c
                            '''

                    if aer_layer_mask[j]:
                         a = tau_aer_wl_layer[i,j] * ssa_aer_wl_layer[i,j] * coefs_aer[0:n_elem,:]

                         gc_layer[i,j,:,0:n_gc_layer[i,j]] += tau_aer_wl_layer[i,j] * ssa_aer_wl_layer[i,j] * coefs_aer[0:n_elem,:] / b

                         if aggregate_gc_derivs:
                             for k in range(n_derivs):
                                 gc_layer_l[i,j,k,:,0:n_coefs_aer] += (tau_aer_wl_layer_l[i,j,k] * ssa_aer_wl_layer  [i,j]   * coefs_aer    [0:n_elem,:] +
                                                                       tau_aer_wl_layer  [i,j  ] * ssa_aer_wl_layer_l[i,j,k] * coefs_aer    [0:n_elem,:] +
                                                                       tau_aer_wl_layer  [i,j  ] * ssa_aer_wl_layer  [i,j]   * coefs_aer_l[k,0:n_elem,:]) / b - \
                                                                       \
                                                                       a * \
                                                                       d[k] / c
                                 '''
                                 gc_layer_l[i,j,k,:,0:n_coefs_aer] += (tau_aer_wl_layer_l[i,j,k] * ssa_aer_wl_layer  [i,j]   * coefs_aer  [i,j,0:n_elem,:] +
                                                                       tau_aer_wl_layer  [i,j  ] * ssa_aer_wl_layer_l[i,j,k] * coefs_aer  [i,j,0:n_elem,:] +
                                                                       tau_aer_wl_layer  [i,j  ] * ssa_aer_wl_layer  [i,j]   * coefs_aer_l[i,j,k,0:n_elem,:]) / b - \
                                                                       \
                                                                       a * \
                                                                       d[k] / c
                                 '''

        return tau_layer_wl, tau_layer_wl_l, ssa_layer_wl, ssa_layer_wl_l, n_gc_layer, gc_layer, gc_layer_l


    ############################################################################
    # Calculate intensities for a single band.
    ############################################################################
    def intensity(self,
                  i_band,
                  tau_gas_wl_layer,
                  tau_gas_wl_q,
                  tau_gas_wl_co2,
                  tau_gas_wl_ch4,
                  tau_ray_wl_layer,
                  aerosol_layer_mask,
                  tau_aerosol_wl,
                  ssa_aerosol_wl,
                  P_aerosol,
                  qext_aerosol_band_0,
                  qext_aerosol,
                  albedo,
                  band_solar_irradiance):

#       print('<<<<<<<<<<<<<<<<<<<< xrtm rt')

        n_chans  = tau_gas_wl_layer.shape[0]
        n_layers = tau_gas_wl_layer.shape[1]
        n_derivs = 5
        n_elem   = 1

        # Dealing with divide by zero issues
        if qext_aerosol_band_0[0] == 0:
            qext_scaling = np.zeros((len(qext_aerosol)))
        else:
            qext_scaling = qext_aerosol/qext_aerosol_band_0[0]

        tau_gas_wl_layer_l = np.zeros((n_chans, n_layers, n_derivs))
        tau_gas_wl_layer_l[:,:,0] = 1.
        tau_gas_wl_layer_l[:,:,1] = 1.
        tau_gas_wl_layer_l[:,:,2] = 1.

        tau_ray_wl_layer_l = np.zeros((n_chans, n_layers, n_derivs))

        n_coefs_ray = 3
        coefs_ray   = self.rayleigh_coefs(rho_rayleigh)
        coefs_ray_l = np.zeros((                   n_derivs, n_elem, n_coefs_ray))
#       coefs_ray_l = np.zeros((n_chans, n_layers, n_derivs, n_elem, n_coefs_ray))

        tau_aerosol_wl_layer   = np.zeros((n_chans, n_layers))
        tau_aerosol_wl_layer_l = np.zeros((n_chans, n_layers, n_derivs))

        if tau_aerosol_wl is None:
            aerosol_layer_mask[:] = False

        if tau_aerosol_wl is not None:
            tau_aerosol_d_n_lay = tau_aerosol_wl * qext_scaling / aerosol_layer_mask.sum()

            for j in range(n_layers):
                if aerosol_layer_mask[j]:
                    tau_aerosol_wl_layer  [:,j]  += tau_aerosol_d_n_lay
                    tau_aerosol_wl_layer_l[:,j,4] = qext_scaling

        ssa_aerosol_wl_layer   = np.zeros((n_chans, n_layers))
        ssa_aerosol_wl_layer_l = np.zeros((n_chans, n_layers, n_derivs))

        if tau_aerosol_wl is not None:
            for j in range(n_layers):
                if aerosol_layer_mask[j]:
                    ssa_aerosol_wl_layer[:,j] += ssa_aerosol_wl

        n_coefs_aerosol, \
        coefs_aerosol = self.aerosol_coefs(i_band)
        coefs_aerosol_l = np.zeros((                   n_derivs, n_elem, n_coefs_aerosol))
#       coefs_aerosol_l = np.zeros((n_chans, n_layers, n_derivs, n_elem, n_coefs_aerosol))

        tau_layer_wl, tau_layer_wl_l, ssa_layer_wl, ssa_layer_wl_l, n_gc_layer, gc_layer, gc_layer_l = \
            self.aggregate_optical_props(tau_gas_wl_layer,
                                         tau_gas_wl_layer_l,
                                         tau_ray_wl_layer,
                                         tau_ray_wl_layer_l,
                                         n_elem,
                                         n_coefs_ray,
                                         coefs_ray,
                                         coefs_ray_l,
                                         aerosol_layer_mask,
                                         tau_aerosol_wl_layer,
                                         tau_aerosol_wl_layer_l,
                                         ssa_aerosol_wl_layer,
                                         ssa_aerosol_wl_layer_l,
                                         n_coefs_aerosol,
                                         coefs_aerosol,
                                         coefs_aerosol_l)

        albedo_l = np.zeros((n_derivs))
        albedo_l[3] = 1.


        I         = np.zeros((n_chans))           # wn
        I_q       = np.zeros((n_chans, n_layers)) # wn x layers
        I_co2     = np.zeros((n_chans, n_layers)) # wn x layers
        I_ch4     = np.zeros((n_chans, n_layers)) # wn x layers
        I_albedo  = np.zeros((n_chans))           # wn
        I_aerosol = np.zeros((n_chans))           # wn


        model = self.call_xrtm_init(np.amax(n_gc_layer), tau_layer_wl[0], tau_layer_wl_l[0])

        for i in range(n_chans):
            I_p, I_m, K_p, K_m = \
                self.call_xrtm_radiance(i,
                                        model,
                                        tau_layer_wl[i],
                                        tau_layer_wl_l[i],
                                        ssa_layer_wl[i],
                                        ssa_layer_wl_l[i],
                                        n_gc_layer[i],
                                        gc_layer[0],
                                        gc_layer_l[0],
                                        albedo,
                                        albedo_l)

            I[i]         = I_p[0,0,0,0]
            I_q[i,:]     = K_p[0,0,0,0,0] * tau_gas_wl_q[i,:] / 10.
            I_co2[i,:]   = K_p[0,1,0,0,0] * tau_gas_wl_co2[i,:] / 10.
            I_ch4[i,:]   = K_p[0,2,0,0,0] * tau_gas_wl_ch4[i,:] / 10.
            I_albedo[i]  = K_p[0,3,0,0,0]
            I_aerosol[i] = K_p[0,4,0,0,0]

        del model


        I         = I         * band_solar_irradiance
        I_q       = I_q       * band_solar_irradiance
        I_co2     = I_co2     * band_solar_irradiance
        I_ch4     = I_ch4     * band_solar_irradiance
        I_albedo  = I_albedo  * band_solar_irradiance
        I_aerosol = I_aerosol * band_solar_irradiance


        return I, I_albedo, I_aerosol, I_q, I_co2, I_ch4


    ############################################################################
    # Create an XRTM instance.
    ############################################################################
    def call_xrtm_init(self, max_coef, tau_layer, tau_layer_l):

        # See documentation for options
        options       = ['calc_derivs', 'delta_m', 'n_t_tms', 'output_at_levels',
                         'sfi', 'source_solar']

        # If using more than two streams use the 4 or 6 stream analytical solvers
        # or one of the n-stream solvers such as 'eig_bvp'.
        solvers       = ['two_stream']

        n_quad        = 1			# Number of polar quadture points
                                                # (# of streams = 2 * n_quad)
        n_stokes      = 1			# Number of stokes elements
        n_derivs      = tau_layer_l.shape[1]	# Number of derivatives
        n_layers      = tau_layer  .shape[0]	# Number of layers
        n_theta_0s    = 1			# Not currently used
        n_kernel_quad = 16			# Number of BRDF azimutal quadrature points
        kernels       = ['lambertian']		# BRDF kernels to use
        n_out_levels  = 1			# Number of levels at which to output radiances
        n_out_thetas  = 1			# Number of zenith angles at which to output radiances


        try:
            model = xrtm.xrtm(options, solvers, max_coef, n_quad, n_stokes, n_derivs,
                    n_layers, n_theta_0s, n_kernel_quad, kernels, n_out_levels, n_out_thetas)
        except xrtm.error as e:
            print(str(e) + '\nERROR: xrtm.init()')
            exit()

        return model


    ############################################################################
    # Calculate intensities for a single band.
    ############################################################################
    def call_xrtm_radiance(self,
                           i_point,
                           model,
                           tau_layer,
                           tau_layer_l,
                           ssa_layer,
                           ssa_layer_l,
                           n_gc_layer,
                           gc_layer,
                           gc_layer_l,
                           albedo,
                           albedo_l):

        #***********************************************************************
        # Define inputs.
        #***********************************************************************
        solver        = 'two_stream'

        n_elem        = 1

        F_0           = 1.		# Downward isotropic emission source at TOA
        theta_0       = self.sza_0	# Solar zenith angle
        phi_0         = 0.		# Solar azimuth angle

        out_levels    = [0]		# Output at TOA

        out_thetas    = [ self.sza]	# Output satellite zenith angle
        out_phis      = [[self.phi]]	# Output satellite azimuth angle


        #***********************************************************************
        # Set inputs.
        #***********************************************************************
        try:
            model.set_fourier_tol(.0001)
        except xrtm.error as e:
            print(str(e) + '\nERROR: xrtm.set_fourier_tol()')
            exit()

        # Levels at which to output radiances
        try:
            model.set_out_levels(out_levels)
        except xrtm.error as e:
            print(str(e) + '\nERROR: xrtm.set_out_levels()')
            exit()

        # The viewing angles at which to output radiances
        try:
            model.set_out_thetas(out_thetas)
        except xrtm.error as e:
            print(str(e) + '\nERROR: xrtm.set_out_thetas()')
            exit()

        # Downward isotropic emission source at TOA
        try:
            model.set_F_iso_top(0.)
        except xrtm.error as e:
            print(str(e) + '\nERROR: xrtm.set_F_iso_top()')
            exit()

        # Upward isotropic emission source at BOA
        try:
            model.set_F_iso_bot(0.)
        except xrtm.error as e:
            print(str(e) + '\nERROR: xrtm.set_F_iso_bot()')
            exit()

        # Solar irradiance at TOA
        try:
            model.set_F_0(F_0)
        except xrtm.error as e:
            print(str(e) + '\nERROR: xrtm.set_F_0()')
            exit()

        # Solar zenith angle
        try:
            model.set_theta_0(theta_0)
        except xrtm.error as e:
            print(str(e) + '\nERROR: xrtm.set_theta_0()')
            exit()

        # Solar azimuth angle
        try:
            model.set_phi_0(phi_0)
        except xrtm.error as e:
            print(str(e) + '\nERROR: xrtm.set_phi_0()')
            exit()

        #***********************************************************************
        # Set optical property inputs.
        #***********************************************************************

        # Optical thickness for each layer (n_layers)
        try:
            model.set_ltau_n(tau_layer)
        except xrtm.error as e:
            print(str(e) + '\nERROR: xrtm.set_ltau_n()')
            exit()

        # Single scattering albed for each layer (n_layers)
        try:
            model.set_omega_n(ssa_layer)
        except xrtm.error as e:
            print(str(e) + '\nERROR: xrtm.set_omega_n()')
            exit()

        # Phase matrix expansion coefficients for each layer
        # (n_layers, n_elem, n_coefs)
        try:
            model.set_coef_n(n_gc_layer, gc_layer[:,0:n_elem,:])
        except xrtm.error as e:
            print(str(e) + '\nERROR: xrtm.set_coef_n()')
            exit()

        # Surface albedo
        try:
            model.set_kernel_ampfac(0, albedo)
        except xrtm.error as e:
            print(str(e) + '\nERROR: xrtm.set_kernel_ampfac()')
            exit()


        #***********************************************************************
        # Set linearised inputs for each layer and each derivative.
        #***********************************************************************
        try:
            model.set_ltau_l_nn(tau_layer_l) # (n_layers, n_derivs)
        except xrtm.error as e:
            print(str(e) + '\nERROR: xrtm.set_ltau_l_nn()')
            exit()

        try:
            model.set_omega_l_nn(ssa_layer_l) # (n_layers, n_derivs)
        except xrtm.error as e:
            print(str(e) + '\nERROR: xrtm.set_omega_l_nn()')
            exit()

        try:
            model.set_coef_l_nn(gc_layer_l[:,:,0:n_elem,:]) # (n_layers, n_derivs, n_elem, n_coefs)
        except xrtm.error as e:
            print(str(e) + '\nERROR: xrtm.set_coef_l_nn()')
            exit()

        try:
            model.set_kernel_ampfac_l_n(0, albedo_l) # (n_derivs)
        except xrtm.error as e:
            print(str(e) + '\nERROR: xrtm.set_kernel_ampfac_l_n()')
            exit()

        try:
            model.update_varied_layers()
        except xrtm.error as e:
            print(str(e) + '\nERROR: model.update_varied_layers()')
            exit()


        #***********************************************************************
        # Run the model for radiances and associated derivatives.
        #
        # I_p/m (up/down radiance)    is (n_levels, n_thetas,           n_phis, n_stokes)
        # K_p/m (up/down derivatives) is (n_levels, n_thetas, n_derivs, n_phis, n_stokes)
        #***********************************************************************
        try:
            I_p, I_m, K_p, K_m = model.radiance(solver, len(out_phis[0]), out_phis)
        except xrtm.error as e:
            print(str(e) + '\nERROR: model.radiance()')
            exit()


        #***********************************************************************
        # Output results at a couple points for development testing.
        #***********************************************************************
#       if i_point == 0 or i_point == 5000:
#           for i in range(0, model.get_n_out_levels()):
#               print('level: %d' % i)
#               print('     intensity:')

#               for j in range(0, model.get_n_out_thetas()):
#                   print('          theta = %9.2E, I_p = %13.6E, I_m = %13.6E' % \
#                         (out_thetas[j], I_p[i,j,0,0], I_m[i,j,0,0]))
#               for j in range(0, model.get_n_derivs()):
#                   print('     derivative: %d' % j)
#                   for k in range(0, model.get_n_out_thetas()):
#                       print('          theta = %9.2E, K_p = %13.6E, K_m = %13.6E' % \
#                             (out_thetas[k], K_p[i,j,k,0,0], K_m[i,j,k,0,0]))
#           print()


        #***********************************************************************
        # Delete xrtm instance.
        #***********************************************************************
        del model


        return I_p, I_m, K_p, K_m
