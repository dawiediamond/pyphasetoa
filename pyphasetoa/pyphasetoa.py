"""Main module."""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.signal import hilbert

class bttSignal():
    """ This class is used to create a BTT signal from an analogue tachometer signal and 
    an analogue proximity probe signal.
    """
    def __init__(self, tacho, dt_tacho, probe, dt_probe, N=1):
        """
        tacho : A 1D numpy array containing the analogue signal of the tachometer.

        dt_tacho : float. The space between consecutive samples in the tachometer, i.e. the inverse of the sampling rate.

        probe : A 1D numpy array containing the analogue signal of the probe.

        dt_tacho : float. The space between consecutive samples in the probe signal, i.e. the inverse of the sampling rate.

        N : The number of encoder sections in the tachometer.

        Note that the first samples of tacho and probe needs to coincide with one another. The end does not need to coincide.

        """
        self.tacho = pd.Series(tacho, index= np.arange( len(tacho) ) * dt_tacho,name='tacho')
        self.dt_tacho = dt_tacho
        self.probe = pd.Series(probe, index= np.arange( len(probe) ) * dt_probe, name='probe')
        self.dt_probe = dt_probe
        self.N = N
        self.tacho_ref = None
        self.pulse_locations = None
        
        self.filt = None
        self.filt_width = None
        self.filt_theta = None
        self.d_theta = None
    
    def info(self):
        """ A function used to print general information about the raw probe and tacho signals.
        """
        return {
            "probe_volt_max":self.probe.max(),
            "probe_volt_min":self.probe.min(),
            "probe_time_max": self.probe.index.max(),
            "tacho_volt_max":self.tacho.max(),
            "tacho_volt_min":self.tacho.min(),
            "tacho_time_max": self.tacho.index.max(),
        }
    
    def plot(self, min_time=None, max_time=None, decimation = 1, markers=False):
        """ A function used to plot the tacho and probe signal on the same axis.

            max_time : The maximum time value to plot up to.

            decimation : The spacing between the samples. It is recommended to start decimation high, maybe at 20, and then take it down gradually.        
        """
        info = self.info()
        
        if max_time is None:
            max_time = max(info['probe_time_max'], info['tacho_time_max'])
        if min_time is None:
            min_time = 0

        ix_tacho = (self.tacho.index < max_time) & (self.tacho.index >= min_time)
        ix_probe = (self.probe.index < max_time) & (self.probe.index >= min_time)
        
        plt.plot(self.tacho.index[ix_tacho].values[::decimation],self.tacho[ix_tacho].values[::decimation], color='r', label=f'Tacho signal')

        plt.plot(self.probe.index[ix_probe].values[::decimation],self.probe[ix_probe].values[::decimation],marker='o',color='g', label=f'Probe signal')

        plt.xlabel('Time [s]', fontsize=20)
        plt.xlabel('Signal [V]', fontsize=20)
        plt.legend().set_draggable(True)

        plt.show()

    def plotThetaRevos(self, revos=[0], decimation = 1, markers=False, x_val='theta'):
        """ A function used to plot several revolutions in the order domain over one another.

            revos : List of int. The revos to plot

            decimation : The spacing between the samples. It is recommended to start decimation high, maybe at 20, and then take it down gradually.        
        """
        info = self.info()
        
        for i, j in self.df_ref_groups.groups.items():
            if i in revos:
                if x_val=='theta':
                    plt.plot(self.df_ref.loc[j, 'theta'].values[::decimation],  self.df_ref.loc[j, 'probe'].values[::decimation], 'o-', label=f'Revo={i}')
                else:
                    plt.plot(self.df_ref.loc[j,'original_time'].values[::decimation],  self.df_ref.loc[j, 'probe'].values[::decimation], 'o-', label=f'Revo={i}')

        plt.xlabel('Angke [rad]', fontsize=20)
        plt.xlabel('Signal [V]', fontsize=20)
        plt.legend().set_draggable(True)

        plt.show()
    
    def applyThreshold(self, series, threshold_value=None, gradient=1, N=1):
        """ 
        
        This is the threshold method used to trigger a ToA or zero crossing. 

        series : The series to calculate the ToAs on. This must be a Pandas series

        thresholf_value: The threshold value to use for the series.

        gradient : If 1, trigger will happen on positive slope, if -1, trigger will happend on negative slope.
        
        returns a Pandas series series with triggered ToAs. The index of the series is the order in which this trigger occurred.
        """
        # Get the threshold value if it is not specified.
        if threshold_value is None:
            threshold_value = (series.max() - series.min())/2
        # Get the sign signal
        series_sign = np.sign(series-threshold_value)
        # Get the diff of the sign
        series_diff = series_sign.diff()
        
        ind_diff = np.where(series_diff == 2*gradient)[0]
        t0 = series.index[0]
        dt = series.index[2] - series.index[1]
        #print('t0:', t0)
        zero_cross = pd.Series(ind_diff * dt, name='ToA') + t0
        #print("SERIES RAW:\n", series.iloc[102:107])
        #print(series.iloc[ind_diff] - threshold_value)
        #print("VALUES:", ind_diff, '(1): ', ind_diff-1)
        #print(series.iloc[ind_diff],'\n', series.iloc[ind_diff-1])
        
        interpol_zero_cross = zero_cross - ((series.values[ind_diff] - threshold_value)/(series.values[ind_diff] - series.values[ind_diff-1])*dt)
        return interpol_zero_cross[::N]
    
    def applyConstantFractionCrossing(self, probe_signal, bins_left, bins_right, frac=0.5):
        """ This function is used to trigger obtain the ToA using the constant fraction crossing method.
        """
        positions = {}
        min_value = probe_signal.min()
        for l, r, probe in zip(bins_left, bins_right, np.arange(len(bins_right))):
            signal_bin = probe_signal[ (probe_signal.index >=l) & (probe_signal.index <= r) ]
            max_value = signal_bin.max()
            index_max = np.argmax(signal_bin.values)
            threshold = (max_value - min_value)*frac + min_value
            toa = self.applyThreshold(signal_bin.iloc[index_max:], threshold, -1).iloc[0]
            positions[probe] = toa
        return positions

    def _makeRefSignal(self, decimation=1):
        """ This function is used to create a DataFrame where each signal's reference revolution
        is indicated in a column next to the signal.
        """
        def getTheta(x):
            t_start = x.iloc[0]['t_start']
            t_end = x.iloc[0]['t_end']
            f = interp1d([t_start, t_end],[0,2*np.pi])
            return f(x['original_time'])

        self.df_ref = pd.DataFrame(self.probe[::decimation])

        #self.df_ref['revo'] = 0
        #self.df_ref['t_start'] = 0
        # Now calculate the tachometer signal if it has not been calculated already.
        index_grouped = pd.cut( pd.Series(self.df_ref.index),
          self.tacho_ref.values,
           right=False,
           retbins=False,
           labels=self.tacho_ref.values[:-1]).astype(float)
        t_end_grouped = pd.cut( pd.Series(self.df_ref.index),
          self.tacho_ref.values,
           right=False,
           retbins=False,
           labels=self.tacho_ref.values[1:]).astype(float)
        revos_grouped = pd.cut( pd.Series(self.df_ref.index),
            self.tacho_ref.values,
            right=False,
            retbins=False,
            labels=np.arange(len(self.tacho_ref.values)-1)).astype(float)
        self.df_ref['t_start'] = index_grouped.values
        self.df_ref['t_end'] = t_end_grouped.values
        self.df_ref['revo'] = revos_grouped.values
        self.df_ref = self.df_ref.dropna()
        self.df_ref['revo'] = self.df_ref['revo'].astype(int)
        self.df_ref['original_time'] =self.df_ref.index
        self.df_ref.index = self.df_ref.index - self.df_ref['t_start']
        theta_order_tracked = []
        for i, j in self.df_ref.groupby('revo').groups.items():
            theta_order_tracked.extend( getTheta( self.df_ref.loc[j] ) )
        self.df_ref['theta'] = theta_order_tracked

        self.df_ref_groups = self.df_ref.groupby('revo')
    
    def _getPulseLocations(self, threshold=0.25, extend_theta=True):
        """ Function used to find pulse locations using the first revolution of the shaft.
            
            threshold : The threshold value to use to determine the pulse locations

            extend_theta : Whether or not to extend the theta_left and theta_right by (theta_right-theta_left)/2 to either side.
        """

        info = self.info()
        threshold_value = (info['probe_volt_max'] - info['probe_volt_min']) * threshold + info['probe_volt_min']
        
        ix = self.df_ref['revo'] == 0
        series_for_localisation = pd.Series(self.df_ref.loc[ix, 'probe'].values,
             index=self.df_ref.loc[ix, 'original_time'])
        
        df_locations_left = self.applyThreshold(series_for_localisation,
              gradient=1, threshold_value=threshold_value).rename('ToA_left')
        df_locations_right = self.applyThreshold(series_for_localisation,
              gradient=-1, threshold_value=threshold_value).rename('ToA_right')
        
        self.pulse_locations = pd.concat([df_locations_left, df_locations_right], axis=1)
        self.pulse_locations['ToA_center'] = (self.pulse_locations['ToA_right'] + self.pulse_locations['ToA_left'])/2
    
        first_row = self.df_ref.iloc[0]

        f = interp1d([first_row['t_start'], first_row['t_end']] , [0,2*np.pi])
        theta_left = f(df_locations_left)
        theta_right = f(df_locations_right)
        self.pulse_locations['Theta_left'] = theta_left
        self.pulse_locations['Theta_right'] = theta_right
        self.pulse_locations['Theta_center'] = (self.pulse_locations['Theta_left'] +  self.pulse_locations['Theta_right']) / 2
        if extend_theta:
            width = self.pulse_locations['Theta_right'] - self.pulse_locations['Theta_left']
            self.pulse_locations['Theta_right'] += width
            self.pulse_locations['Theta_left'] -= width

    def calculateX(self, methods=['phase','threshold', 'constant-fraction'], params={}, decimation=1, noise_std=None, verbose=True):
        """ This function calculates the tip deflections using one of five methods. This function is the only function that should get called using this library.

        methods : A list of str. Can contain 'phase', 'threshold', 'constant-fraction', 'maxvalue', 'maxchangerate'. All the methods listed will be used to calculate the tip deflection.

        params : A dictionary specifying the hyperparameters for the the specified method. The format is:
            params = {
                'phase':{
                    filt_width : int, # Defaults to 4
                    rho : float, # Defaults to the average angular distance between the pulses.
                    filtlen : int # Defaults to 6001
                },
                'threshold':{
                    'threshold_percentage': float, # Defaults to 0.25
                    'threshold_value': float # This value takes presedence if it is specified. If not specified, use threshold percentage,
                    'gradient': 1 or -1 # This value defaults to 1
                },
                'constant-fraction':{
                    'frac' : between 0 and 1 # Defaults to 1
                }
            }

        decimation : Int. The amount that the probe signal must be decimated in the time domain. If 2, then every second value is retained, if 4 then every 4th value is retained and so on and so forth. 

        verbose: Bool. A setting that allows for the printing of infomation during the analysis.

        noise_std : positive float. This is the standard deviation of the noise you want to add to the proximity probe signal. The noise is normally distributed with a mean of 0.

        returns A Pandas DataFrame containing all the ToAs and a Series containing all the t_start values
        """
        # First determine if the pulse locations have been determined thus far, if not, calculate them
        if self.pulse_locations is None:
            if self.tacho_ref is None:
                self.tacho_ref = self.applyThreshold(self.tacho, N=self.N)
            self._makeRefSignal()
            self._getPulseLocations()
        info = self.info()
        # First decimate the results
        self._makeRefSignal(decimation=decimation)

        if 'phase' in methods:
            phase_params = params.get('phase', {})
            phase_params.get('rho')
            phase_params.get('filtlen')
            phase_params.get('filtwidth')

            self._genFilt(phase_params.get('rho', None),
                          phase_params.get('filtlen', 6001),
                          phase_params.get('filtwidth', 4))
            
            filt_locations = {}
            filt_locations_1 = {}

            for i in self.pulse_locations.index:
                theta_left = self.pulse_locations.loc[i]['Theta_center']
                filt_locations[i] = theta_left + self.filt_theta
                filt_locations_1[i] = theta_left + self.filt_theta - self.d_theta
            phase_results = []
            phase_results_1 = []

        if 'threshold' in methods:
            threshold_params = params.get('threshold', {})
            
            threshold_percentage = threshold_params.get('threshold_percentage', 0.25)
            if not 'threshold_value' in threshold_params:
                threshold_value =  (info['probe_volt_max'] - info['probe_volt_min']) * threshold_percentage + info['probe_volt_min']
            else:
                threshold_value = threshold_params.get('threshold_value')
            
            threshold_gradient = threshold_params.get('gradient', 1)

            threshold_results = []
        
        if 'constant-fraction' in methods:
            cf_params = params.get('constant-fraction', {})
            cf_frac = cf_params.get('frac', 0.5)

            cf_results = []
        index = []
        for revo in tqdm(self.df_ref.revo.unique()):
            index.append(f"Rev {revo}")
            
            df_revo = self.df_ref[self.df_ref['revo']==revo]
            
            # Determine the out of bounds value
            out_of_bounds_value = (lambda x: np.median(x[:int(len(x)/10)]  )) ( np.sort(df_revo['probe'].copy(deep=True).values) )
            if noise_std is None:
                # No noise added
                f_interp = interp1d(df_revo['theta'],  df_revo['probe'], fill_value=out_of_bounds_value, bounds_error=False)
            else:
                # Noise added
                f_interp = interp1d(df_revo['theta'], df_revo['probe'] + noise_std * np.random.randn(len(df_revo['probe'])), fill_value=out_of_bounds_value, bounds_error=False)
            
            if 'phase' in methods:
                pulse = {}
                pulse_1 = {}

                for i in self.pulse_locations.index:
                    probe_signal = f_interp(filt_locations[i])
                    probe_signal_1 = f_interp(filt_locations_1[i])

                    pulse[i] = (lambda x: x.sum()) (probe_signal * self.filt)
                    pulse_1[i] = (lambda x: x.sum()) (probe_signal_1 * self.filt)

                phase_results.append(pulse)
                phase_results_1.append(pulse_1)
            
            if 'threshold' in methods:
                d_theta = df_revo['theta'].diff().min()
                new_x = np.arange(0,2*np.pi, d_theta/2)
                threshold_calc_series = pd.Series(f_interp(new_x), index = new_x)
                # Now ensure you pick the first value from each bin
                result = self.applyThreshold(threshold_calc_series,threshold_value=threshold_value,gradient=threshold_gradient)
                result_dict = {}
                for left, right, count in zip(self.pulse_locations['Theta_left'].tolist(), self.pulse_locations['Theta_right'].tolist(), np.arange(len(self.pulse_locations['Theta_left']))):
                    result_dict[count] = result[(result > left) & (result < right)].iloc[0]                
                threshold_results.append(result_dict)
            
            if 'constant-fraction' in methods:
                if not 'threshold' in methods:
                    d_theta = df_revo['theta'].diff().min()
                    new_x = np.arange(0,2*np.pi, d_theta/2)
                    threshold_calc_series = pd.Series(f_interp(new_x), index = new_x)
                
                result = self.applyConstantFractionCrossing(threshold_calc_series,
                                self.pulse_locations['Theta_left'].tolist(),
                                self.pulse_locations['Theta_right'].tolist(),
                                frac=cf_frac)

                cf_results.append(result)
        return_dict = {}

        if 'phase' in methods:
            # Now do phase results
            df_result = pd.DataFrame(phase_results, index=index)
            df_result_1 = pd.DataFrame(phase_results_1, index=index)
            df_phase = pd.DataFrame(np.angle(df_result.values), index=index)
            df_phase_1 = pd.DataFrame( np.angle(df_result_1.values), index=index)
            # Convert phase angles to 0  - 2pi
            df_phase[df_phase < 0] += 2*np.pi
            df_phase_1[df_phase_1 < 0] += 2*np.pi
            # Get dphase_dt
            dphase_dtheta = (df_phase - df_phase_1)/self.d_theta
            # Get dphase_revo
            dphase_drevo = df_phase - df_phase.iloc[0]
            X_Phase = (dphase_drevo / dphase_dtheta) * 1000000
            return_dict['phase'] = X_Phase.iloc[1:, :]
        
        if 'threshold' in methods:        
            # Do threshold results
            df_threshold_results = pd.DataFrame(threshold_results, index=index)
            X_threshold = -1*(df_threshold_results - df_threshold_results.iloc[0]) * 1000000
            return_dict['threshold'] = X_threshold.iloc[1:, :]
        
        if 'constant-fraction' in methods:
            # Do constant fraction crossing results
            df_cf = pd.DataFrame(cf_results, index=index)
            X_cf = -1*(df_cf - df_cf.iloc[0]) * 1000000
            return_dict['constant-fraction'] = X_cf.iloc[1:, :]
        return return_dict

    def _genFilt(self, rho=None, filtlen=6001, filtwidth=4, verbose=False):
        """ A funtion used to generate a complex filter.

        rho : The parameter of the filter, this governs the width of the filter. If no value is specified, the width
        of the pulse in self.pulse_locations is used.
        
        filtlen : The number of points that will be present in the filter.

        filtwidth : Integer. The width of the filter will be filtwidth*rho
        
        """
        def genHilbert(rho, theta):
            G2 = -2 * np.exp(-1 * theta**2/rho**2 ) * (1/rho**2 - 2 * theta**2/rho**4)
            K2 = hilbert(G2)
            return K2 / K2.max()
        
        if rho is None:
            self.rho = (self.pulse_locations['Theta_right'] - self.pulse_locations['Theta_left']).mean()
        else:
            self.rho = rho

        if verbose:
            print('---Generating filter with rho={self.rho}, width={self.filt_width} * rho and length {filtlen}')
        self.filt_width = filtwidth
        self.filt_theta = np.linspace(-self.rho*self.filt_width, self.rho*self.filt_width, filtlen)
        self.d_theta = self.filt_theta[2] - self.filt_theta[1]
        self.filt = genHilbert(self.rho, self.filt_theta)
    
    def plotFilt(self, show=False):
        """ This function plots the filter
        """
        plt.plot(self.filt_theta, self.filt.real, label='Real')
        plt.plot(self.filt_theta, self.filt.imag, label='Imag')
        plt.xlabel('Angle [rad]', fontsize=20)
        plt.xlabel('Filter intensity', fontsize=20)
        plt.title(r'Complex filter, $\rho$='+str(round(self.rho, 4))+', width='+str(self.filt_width)+r'$\rho$')
        plt.legend().set_draggable(True)
        plt.grid(True)
    
    def getRevo(self, revo):
        """ A function used to return a revolution's data. 
        """
        return self.df_ref[self.df_ref['revo'] == revo]


# Get the ToAs using constant fraction crossing, threshold, max value and/or max change.
# Get the ToAs using phase based method.
# Observe results in Fourier form or another.