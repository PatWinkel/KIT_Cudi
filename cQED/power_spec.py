# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:10:27 2016

@author: Patrick
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as col
from qkit.storage import hdf_lib
from qkit.analysis.circle_fit.circuit import reflection_singleport as rf_p
from cpw_resonator import cpw_resonator as cpw
import scipy.constants as cs
import os
import math as mt
from scipy.ndimage import gaussian_filter1d
 
class power_spec(object):
    '''
    contains all functions to create a power spectrum plot
    '''
    
    def __init__(self, filepath, colorset = 'default', save = False, datafile = True, h5 = True, gaussfilter = False):
        '''
        optional settings: 'save' = True to save generated plots, 'datafile': filepath or data array
        'h5': file format
        '''

        self._save          =       save  
        self._dpi           =       400   
        self._attenuation   =       0     
        self._attenuation_mem =     0
        self._tolerance     =       1
        self._data_real_gen = []
        self._data_imag_gen = []
        self._gauss_number = 3.
        self._width = 15
        self._height = 7
        
        #extracting data set:
        if datafile:  
                                                         #if filepath is a real path to a real file, take the data
            self._filepath      =       filepath
            self._name          =       self._file_name(filepath)
            if h5:
                
                self._data          =       hdf_lib.Data(path  =  filepath)
                self._amp           =       np.array(self._data['/entry/data0/amplitude'])
                self._phase         =       np.array(self._data['/entry/data0/phase'])
                self._freq          =       np.array(self._data['/entry/data0/frequency'])
                try:
                    self._power         =       np.array(self._data['/entry/data0/power (dbm)'])
                except KeyError:
                    self._power         =       np.array(self._data['/entry/data0/power'])              
                except KeyError:
                    print 'could not load power data'
                    self._power = [0.0]

                    
            
            else:                                                       #if dataset is comprised in .dat file
                self._data = np.genfromtxt(filepath).T
                self._power = np.array(np.unique(self._data[0]))
                self._freq = np.array(np.extract(self._data[0] == self._data[0][0], self._data[1]))
                self._amp = []
                for f in self._freq:
                    self._amp.append(np.extract(self._data[1] == f, self._data[2]))
                self._amp = np.array(self._amp).T
                self._phase = []
                for f in self._freq:
                    self._phase.append(np.extract(self._data[1] == f, self._data[3]))
                self._phase = np.array(self._phase).T
            
            self._prep_data(gaussfilter)
            self._set_window_default()
        else:
            self._data          =       filepath                               #it is also possible to pass an array directly to the class instance
            self._name          =       raw_input('filename: ')
        
        #properties (color) of colorbar (2D) & general font settings
        if colorset == 'default':
            self._startcolor        =   'black'
            self._midcolor          =   'blue'
            self._endcolor          =   'white'
            self._font              =    {'weight' : 'normal', 'size' : 22}
            self._labelsize         =    30
        #changing setting by input request
        else:
            print 'colorset non-default'
            self._startcolor        =     raw_input('startcolor: ')
            self._midcolor          =     raw_input('midcolor: ')
            self._endcolor          =     raw_input('endcolor: ')
            self._font_weight       =     raw_input('font weight (exp. normal): ')
            self._font_size         =     float(raw_input('font size: '))
            self._font              =     {'weight' : self._font_weight, 'size' : self._font_size}
            self._labelsize         =     float(input('labelsize:'))
        
        self._c_cmap = col.LinearSegmentedColormap.from_list('olive',[self._startcolor,self._midcolor,self._endcolor], N = 256, gamma = 1.0)
        cm.register_cmap(cmap = self._c_cmap)
        #font = self._font
        plt.rc('font', **self._font)
    
        
    '''set value for attenuation (during measurement)'''
        
    def _set_attenuation(self, attenuation):
        self._attenuation_mem   = self._attenuation         #introducing memory variable to circumvent changes in _power when not neccessary (multiple calls of _set_attenuation for example)
        self._attenuation       = attenuation
        if self._attenuation_mem != self._attenuation:
            self._power               += self._attenuation_mem
            self._power               -=      self._attenuation
            self._set_window_default()
        else:
            print 'no change - attenuation already: {} dB'.format(self._attenuation)
            
    def _set_gauss_nmb(self, number):
        '''function to change the number of neighbours for the gauss fit'''
        self._gauss_number = number
            
    def _set_figsize(self, width, height):
        self._width = width
        self._height = height
        
    def _get_attenuation(self):
        return self._attenuation        
            
            
    ''' returns power value for given column c '''        
    def _get_power(self,c):
        return self._power[c]
        
    '''returns column number for a arbitrary power value'''    
    def _search_power(self, power, precision):        
        for i in range(len(self._power)):
            if np.abs(power - self._power[i]) < precision:
                return 'column number: {}'.format(i)
                
    '''create a filename from the filepath'''
        
    def _file_name(self, filepath):
        filename_ext        =       os.path.basename(filepath)
        return os.path.splitext(filename_ext)[0]

    
    '''prepare (=normalize and mask) dataset '''    
    def _prep_data(self, gaussfilter = False):
        
        
        self._set_attenuation(self._attenuation)
        
        if gaussfilter:
            self._amp = gaussian_filter1d(self._amp,self._gauss_number)
            self._phase = gaussian_filter1d(self._phase,self._gauss_number)
      
        
        self._data_mask_amp       =       self.data_mask(self._tolerance, self._amp)
        self._amp_n               =       self.normalize_amp(self._amp,self._data_mask_amp)  
        self._amp_spec            =       self._spec(self._amp.T,self._data_mask_amp)
        self._amp_spec_n            =       self._spec(self._amp_n,self._data_mask_amp)
        
        self._data_mask_ph        =       self.data_mask_phase(self._phase)
        self._phase_n             =       self.phaseshift(self._phase, self._data_mask_ph)
        self._phase_spec          =       self._spec(self._phase.T,self._data_mask_ph)
        self._phase_spec_n        =      self._spec(self._phase_n,self._data_mask_ph)
        
        self._prepare_f_range(None, None)
        self._get_data_circle()
        
        self._prep_d = True
         
        
    """data_mask and normalization for amplitude data:  """  
    
    def data_mask(self,tolerance,arr):
        m           =       np.array(arr)                                                
        mask_a      =       []
        for c in range(m.shape[0]):         #m.shape[0]: number of entries in the 0th column; m[c] is the cth column of m (entries are measured with the same power)
            lower_th        =       np.mean(m[c]) - tolerance*np.abs(np.mean(m[c]))      #lower threshold; lower_th = 0.0 for tolerance = 1
            upper_th        =       np.mean(m[c]) + tolerance*np.abs(np.mean(m[c]))      #upper threshold: 2*mean value
               #np.any tests if an element is True or False with respect to the declared condition(s)
            mask_a.append(np.any((m[c] < lower_th, m[c] > upper_th),axis=0))   #creates mask array, that contains the information about invalid values in m
        return np.array(mask_a).T                                              #axis = 0 means testing columnwise
                                                                           
    
    def normalize_amp(self,m,data_mask):    #normalize columnwise
        arr         =       np.array(m.T) 
        for c in range(len(arr[0])):    #m_n[0] contains all frequencies
            min_v           =       np.ma.min(np.ma.masked_array(arr, data_mask)[:,c])    #min value of column
            arr[:,c]        =       arr[:,c] - min_v    #set min value (column) to zero; shift all entries by min. value
            max_v           =       np.ma.max(np.ma.masked_array(arr, data_mask)[:,c])    #max value of column
            arr[:,c] /= max_v   #normalize on max value (column)
        return arr
        
    """data mask and phaseshift for phase data:"""
    
    def data_mask_phase(self,phase):   #creates data mask             #.T creates the transpose of the matrix/array which makes it  
        m           =       np.array(phase)                                               #quite convenient to seperate different entries by sequence unpacking 
        mask_a      =       []
        for c in range(m.shape[0]):                            
            lower_th        =       -np.pi      #lower threshold
            upper_th        =       +np.pi      #upper threshold
            mask_a.append(np.any((m[c] < lower_th, m[c] > upper_th),axis=0))   #creates mask array, that contains the information about invalid values in m
                                          #axis = 0 means testing columnwise
        return np.array(mask_a).T     

    

    
    def phaseshift(self,m, mask):   #shift columnwise (c: column)
        phase_arr = np.array(m.T)
        for c in range(len(phase_arr[0])):                                             
            
            phase_arr[:,c] = np.unwrap(phase_arr[:,c])      #unwrap the column: no larger differences than pi (default value)
            
            min_v = np.ma.min(np.ma.masked_array(phase_arr, mask)[:,c])  #min value for column c

            phase_arr[:,c] = phase_arr[:,c] - min_v       
            
        min_vt = np.ma.min(np.ma.masked_array(phase_arr, mask))             #find min value of array
        max_vt = np.ma.max(np.ma.masked_array(phase_arr, mask))             #find max value of array
        delta = max_vt + min_vt
        phase_arr -= delta/2     #add it to column to shift it into symmetric
            
        return phase_arr
    

    '''extract background slope from amplitude signal'''
    
    def _extract_bkgr(self, p, norm = False):
        self._deltax = self._freq[-1] - self._freq[0]
        if norm:
            self._deltay = self._amp_n.T[p][-1] - self._amp_n.T[p][0]
        else:
            self._deltay = self._amp[p][-1] - self._amp[p][0]
        
        return self._deltay/self._deltax   
        
        
    '''creates the masked power spectrum array'''
    def _spec(self,arr, mask):                 
        return np.ma.masked_array(arr,mask)
   
   
   
   
   
   
    '''set and get plot window settings'''   
   
    def _set_window_default(self):
        self._xmin = self._power[0]
        self._xmax = self._power[-1]
        self._ymin = self._freq[0]
        self._ymax = self._freq[-1]
            
            
            
    def _set_window(self, x_min, x_max, y_min, y_max):
        self._xmin = x_min
        self._xmax = x_max
        self._ymin = y_min
        self._ymax = y_max
        
        
        
    def _get_window(self, name = 'default'):
        if name == 'x_min':
            print 'x_min: {} dBm'.format(self._xmin)
        elif name == 'x_max':
            print 'x_max: {} dBm'.format(self._xmax)
        elif name == 'y_min':
            print 'y_min: {} Hz'.format(self._ymin)
        elif name == 'y_max':
            print 'y_max: {} Hz'.format(self._ymax)
        else:
            return (self._xmin, self._xmax, self._ymin, self._ymax)
            
    
    

    
    '''plot function for external usage (datasets without filepath)'''
    
    def plot(self, x_val, y_val, C_set, fre = None):           
        X, Y = np.meshgrid(x_val, y_val)
        #c_cmap = col.LinearSegmentedColormap.from_list('olive',[self._startcolor,self._midcolor,self._endcolor], N = 256, gamma = 1.0)
        #cm.register_cmap(cmap = c_cmap)
        #font = self._font
        #plt.rc('font', **self._font)
        
        self._fig, axes = plt.subplots(figsize=(self._width,self._height))
        p = axes.pcolor(X,Y,C_set, cmap = self._c_cmap)
        cb = self._fig.colorbar(p, ax = axes)
        cb.set_label(r'$\langle\hat n\rangle \rightarrow \arg (S_{11})\,\mathrm{(a.u.)}$', fontsize=self._labelsize)
        if not fre == None:
            axes.plot(x_val, fre, color = 'red', lw = 4)        
        axes.set_xlim(self._xmin, self._xmax)
        axes.set_ylim(self._ymin, self._ymax)
        axes.set_xlabel(r'$P_{\mu w}\,(\mathrm{dBm})$', fontsize=self._labelsize)
        axes.set_ylabel(r'$f_{\mu w}\,(\mathrm{Hz})$', fontsize=self._labelsize)
 
        
        
        
    """plots a whole power spec for a given attenuation""" 
    
    def _plot_amp(self,norm = False, save = False):
        self._save          =       save
        if norm:
            self.plot(self._power,self._freq,self._amp_spec_n)
        else:
            self.plot(self._power,self._freq,self._amp_spec)
        
        if self._save:
            self._name          +=      '_amp'
            self.plot_save(self._name)
            
            
            
    def _plot_phase(self, norm = False, save = False):
        self._save        = save
        if norm:
            self.plot(self._power,self._freq,self._phase_spec_n)
        else:
            self.plot(self._power,self._freq,self._phase_spec)
        
        if self._save:
            self._name          +=      '_phase'
            self.plot_save(self._name)
    '''        
    def _plot_power(self, y_val, y_val_name):
        self._fig, axes = plt.subplots(figsitze = (10,6))
        axes.plot(self._power, y_val, color = 'blue', lw = 2)
        axes.set_xlabel(r'$P_{\mu w}\,(\mathrm{dBm})$', fontsize= self._labelsize)
        axes.set_ylabel(r'${}$'.format(y_val_name), fontsize= self._labelsize)
        axes.legend()
        
    def _plot_1D(self,x_val, x_val_name, y_val, y_val_name):
        self._fig, axes = plt.subplots(figsitze = (10,6))
        axes.plot(x_val, y_val, color = 'blue', lw = 2)
        axes.set_xlabel(r'${}$'.format(x_val_name), fontsize= self._labelsize)
        axes.set_ylabel(r'${}$'.format(y_val_name), fontsize= self._labelsize)
        axes.legend()
    '''    

    '''function to plot single traces'''
    
    def _plot_trace(self, power_val, amp = True, norm = False, legend = True, log = False):
        #font = self._font
        #plt.rc('font', **font)
        
        self._fig, axes = plt.subplots(figsize=(self._width,self._height))
        
        if log:
            amp_n = np.array(10.*np.log10(self._amp_n.T[power_val]))
            amplitude = np.array(10.*np.log10(self._amp[power_val]))
        else:
            amp_n = self._amp_n.T[power_val]
            amplitude = self._amp[power_val]
        
        if amp:
            if norm:
                axes.plot(self._freq,amp_n, 'k', label ='P = {} dBm'.format(mt.ceil(self._power[power_val]*100)/100), color = 'blue', lw = 2)
                #axes.plot(self._freq,fre)
        
            else:
                axes.plot(self._freq,amplitude, 'k', label ='P = {} dBm'.format(mt.ceil(self._power[power_val]*100)/100), color = 'blue', lw = 2)
                #axes.plot(self._freq,fre)
            if log: axes.set_ylabel(r'$\langle\hat n\rangle \rightarrow \arg (S_{11})\,\mathrm{(dB)}$', fontsize= self._labelsize)
            else:   axes.set_ylabel(r'$\langle\hat n\rangle \rightarrow \arg (S_{11})\,\mathrm{(a.u.)}$', fontsize= self._labelsize)
        else:
            if norm:
                axes.plot(self._freq,self._phase_n.T[power_val], 'k', label ='P = {} dBm'.format(mt.ceil(self._power[power_val]*100)/100), color = 'blue', lw = 2)
                #axes.plot(self._freq,fre)
            else:
                axes.plot(self._freq,self._phase[power_val], 'k', label ='P = {} dBm'.format(mt.ceil(self._power[power_val]*100)/100), color = 'blue', lw = 2)
                #axes.plot(self._freq,fre)
            axes.set_ylabel(r'$\arg (S_{11})\,\mathrm{(rad)}$', fontsize= self._labelsize)
            axes.set_ylim(-np.pi,+np.pi)
        if legend:
            axes.legend()
        axes.set_xlim(self._freq[0], self._freq[-1])    
        axes.set_xlabel(r'$f_{\mu w}\,(\mathrm{Hz})$', fontsize= self._labelsize)
        plt.show()
        
    '''function to plot tilted traces'''    
        
    def _plot_trace_tilt(self, power_val, amp = True, norm = False, legend = True):
        #font = self._font
        #plt.rc('font', **font)
        
        self._fig, axes = plt.subplots(figsize=(self._width,self._height))
        m = self._extract_bkgr(power_val, norm = norm)
        if amp:
            if norm:
                c = self._amp_n.T[power_val][0] - m*self._freq[0]
                axes.plot(self._freq,self._amp_n.T[power_val]-m*self._freq-c, 'k', label ='P = {} dBm'.format(self._power[power_val]), color = 'blue', lw = 2)
        
            else:
                c = self._amp[power_val][0] - m*self._freq[0]
                axes.plot(self._freq,self._amp[power_val]-m*self._freq-c, 'k', label ='P = {} dBm'.format(self._power[power_val]), color = 'blue', lw = 2)
        else:
            axes.plot(self._freq,self._phase[power_val], 'k', label ='P = {} dBm'.format(self._power[power_val]), color = 'blue', lw = 2)
        if legend:
            axes.legend()
        axes.set_xlim(self._freq[0], self._freq[-1])        
        axes.set_ylabel(r'$\langle\hat n\rangle \rightarrow \arg (S_{11})\,\mathrm{(a.u.)}$', fontsize= self._labelsize)
        axes.set_xlabel(r'$f_{\mu w}\,(\mathrm{Hz})$', fontsize= self._labelsize)
        plt.show()
         
        
    '''function to save plots'''    
    def plot_save(self, add):
        self._fig.savefig(str(self._name+add), dpi = self._dpi)
        
    

    
    def _set_data_range(self, data):
        '''
        cuts the data array to the positions where f>=f_min and f<=f_max in the frequency-array
        the fit functions are fitted only in this area
        the data in the .h5-file is NOT changed
        '''
        if data.ndim == 1:
            return data[(self._freq >= self._f_min) & (self._freq <= self._f_max)]
        if data.ndim == 2:
            ret_array=np.empty(shape=(data.shape[0],self._fit_freq.shape[0]),dtype=np.float64)
            for i,a in enumerate(data):
                ret_array[i]=data[i][(self._freq >= self._f_min) & (self._freq <= self._f_max)]
            return ret_array
            
    def _prepare_f_range(self,f_min,f_max):
        '''
        prepares the data to be fitted:
        f_min (float): lower boundary
        f_max (float): upper boundary
        '''

        self._f_min = np.min(self._freq)
        self._f_max = np.max(self._freq)

        '''
        f_min f_max do not have to be exactly an entry in the freq-array
        '''
        if f_min:
            for freq in self._freq:
                if freq > f_min:
                    self._f_min = freq
                    break
        if f_max:
            for freq in self._freq:
                if freq > f_max:
                    self._f_max = freq
                    break

        '''
        cut the data-arrays with f_min/f_max and fit_all information
        '''
        self._fit_freq = np.array(self._set_data_range(self._freq))
        self._fit_amp = np.array(self._set_data_range(self._amp))
        self._fit_phase = np.array(self._set_data_range(self._phase))
        self._fit_phase_n = np.array(self._set_data_range(self._phase_n.T))
     

     
    def _get_data_circle(self, fit_all =True, norm = True):
        '''
    calc complex data from amp and pha
    
        '''
        self._fit_all = fit_all
        
            
        if not self._fit_all:
            self._z_data_raw = np.empty((1,self._fit_freq.shape[0]), dtype=np.complex64)

            if self._fit_amp.ndim == 1: 
                if norm: self._z_data_raw[0] = np.array(self._fit_amp*np.exp(1j*self._fit_phase_n),dtype=np.complex64)     
                else:  self._z_data_raw[0] = np.array(self._fit_amp*np.exp(1j*self._fit_phase),dtype=np.complex64)
                  
            else: 
                if norm: self._z_data_raw[0] = np.array(self._fit_amp[-1]*np.exp(1j*self._fit_phase_n[-1]),dtype=np.complex64)
                else: self._z_data_raw[0] = np.array(self._fit_amp[-1]*np.exp(1j*self._fit_phase[-1]),dtype=np.complex64)
            self._data_real_gen.append(self._z_data_raw[0].real)
            self._data_imag_gen.append(self._z_data_raw[0].imag)

        if self._fit_all:
            self._z_data_raw = np.empty((self._fit_amp.shape), dtype=np.complex64)
            for i,a in enumerate(self._fit_amp):
                if norm:
                    self._z_data_raw[i] = self._fit_amp[i]*np.exp(1j*self._fit_phase_n[i])
                else:
                    self._z_data_raw[i] = self._fit_amp[i]*np.exp(1j*self._fit_phase[i])
                self._data_real_gen.append(self._z_data_raw[i].real)
                self._data_imag_gen.append(self._z_data_raw[i].imag)
        
    def _circle_fit(self, coupling):
        '''
        circle fit routine: calculates res. freq, Quality factor, chi_square, number of photons (n_ph)
        '''
        self._fres_cf = []
        self._fres_pf = []
        self._chi_sqr = []
        self._Qload_cf = []
        self._Qload_pf = []
        self._n_ph = []
        self._z_data_sim = []
        for i in range(len(self._power)):
            cf = rf_p(self._fit_freq, self._z_data_raw[i])
            cf.autofit(coupling, ignoreslope = False, refine_res = True,  plot = True)
            dphase_gauss = cf._derivation(self._fit_freq, self._z_data_raw[i]) 
            _fr = self._freq[np.argmin(dphase_gauss)]            
            theta0, Ql, fr, slope = cf._phase_fit_wslope(self._fit_freq, self._z_data_raw[i], 0, 1000, _fr, 0.)
            self._z_data_sim.append(cf.z_data_sim)
            self._fitresults = cf.fitresults
            self._fres_cf.append(self._fitresults['fr'])
            self._fres_pf.append(fr)
            self._Qload_cf.append(self._fitresults['Ql'])
            self._Qload_pf.append(Ql)
            self._chi_sqr.append(self._fitresults['chi_square'])
            self._n_ph.append(cf.get_photons_in_resonator(self._power[i]))
            
    def _cubic_fit(self, fit = 'cf'):
        if fit == 'cf':
            a, b, c, d = np.polyfit(self._n_ph-self._n_ph[0], self._fres_cf-self._fres_cf[0], 3)
            delta = self._n_ph-self._n_ph[0]
            self._fig, axes = plt.subplots(figsize = (15,7))
            axes.plot(self._n_ph, (self._fres_cf-self._fres_cf[0])/1000., label = r'data (circle fit)', lw = 4, color = 'blue')
            axes.plot(self._n_ph, (a*delta**3.+b*delta**2.+c*delta+d)/1000., label = 'fit', lw = 4, color = 'red' )
            axes.set_xlabel(r'number of photons $n_{\Gamma}$', fontsize = self._labelsize)
            axes.set_ylabel(r'freq. shift $\Delta f$ [kHz]', fontsize = self._labelsize)
            axes.legend()
        else:
            a, b, c, d = np.polyfit(self._n_ph-self._n_ph[0], self._fres_pf-self._fres_pf[0], 3)
            delta = self._n_ph-self._n_ph[0]
            self._fig, axes = plt.subplots(figsize = (15,7))
            axes.plot(self._n_ph, (self._fres_pf-self._fres_pf[0])/1000., label = r'data (phase fit)', lw = 4, color = 'blue')
            axes.plot(self._n_ph, (a*delta**3.+b*delta**2.+c*delta+d)/1000., label = 'fit', lw = 4, color = 'red' )
            axes.set_ylabel(r'freq. shift $\Delta f$ [kHz]', fontsize = self._labelsize)
            axes.set_xlabel(r'number of photons $n_{\Gamma}$', fontsize = self._labelsize)
            axes.legend()
        return a, b, c, d
        
     
    '''useful functions to create merged datasets'''
     
     
    def _compare(self,arr1,arr2):         #compares two arrays for identical entries and creates a boolian mask
        comp_mask = []                    #shape of mask is given by first argument (array)
        comp_mask = np.in1d(arr1,arr2)
        return np.array(comp_mask)
    
    
    
    def _power_add(self,arr1,arr2):       #compares two arrays for identical entries and merges them 
        m = []
        mask_p = []
        if np.min(arr1) > np.min(arr2):
            mask_p = self._compare(arr1,arr2)
            m = np.ma.masked_array(arr1,mask_p).compressed()                                  #compressed() removes all True entries
            m_p = np.append(arr2,m)   #append masked entries of arr1 to arr2
        else:
            mask_p = self._compare(arr2,arr1)
            m = np.ma.masked_array(arr2,mask_p).compressed()
            m_p = np.append(arr1,m)   #append masked entries of arr2 to arr1
        return m_p
        
class time_spec(object):
    '''contains all functions to create a time spectrum plot'''
    
    def __init__(self, filepath, colorset = 'default', save = False, datafile = True, h5 = True):
        '''
        optional settings: 'save' = True to save generated plots, 'datafile': filepath or data array
        'h5': file format
        '''

        self._save          =       save  
        self._dpi           =       400   
        self._attenuation   =       0     
        self._attenuation_mem =     0
        self._tolerance     =       1
        self._data_real_gen = []
        self._data_imag_gen = []
        
        #extracting data set:
        if datafile:  
                                                         #if filepath is a real path to a real file, take the data
            self._filepath      =       filepath
            self._name          =       self._file_name(filepath)
            if h5:
                
                self._data          =       hdf_lib.Data(path  =  filepath)
                self._amp           =       np.array(self._data['/entry/data0/amplitude'])
                self._phase         =       np.array(self._data['/entry/data0/phase'])
                self._freq          =       np.array(self._data['/entry/data0/frequency'])
                self._time      =       np.array(self._data['/entry/data0/time'])

                    
            
            else:                                                       #if dataset is comprised in .dat file
                self._data = np.genfromtxt(filepath).T
                self._power = np.array(np.unique(self._data[0]))
                self._freq = np.array(np.extract(self._data[0] == self._data[0][0], self._data[1]))
                self._amp = []
                for f in self._freq:
                    self._amp.append(np.extract(self._data[1] == f, self._data[2]))
                self._amp = np.array(self._amp).T
                self._phase = []
                for f in self._freq:
                    self._phase.append(np.extract(self._data[1] == f, self._data[3]))
                self._phase = np.array(self._phase).T
            
            self._prep_data()
            self._set_window_default()
        else:
            self._data          =       filepath                               #it is also possible to pass an array directly to the class instance
            self._name          =       raw_input('filename: ')
        
        #properties (color) of colorbar (2D) & general font settings
        if colorset == 'default':
            self._startcolor        =   'black'
            self._midcolor          =   'blue'
            self._endcolor          =   'white'
            self._font              =    {'weight' : 'normal', 'size' : 22}
            self._labelsize         =    30
        #changing setting by input request
        else:
            print 'colorset non-default'
            self._startcolor        =     raw_input('startcolor: ')
            self._midcolor          =     raw_input('midcolor: ')
            self._endcolor          =     raw_input('endcolor: ')
            self._font_weight       =     raw_input('font weight (exp. normal): ')
            self._font_size         =     float(raw_input('font size: '))
            self._font              =     {'weight' : self._font_weight, 'size' : self._font_size}
            self._labelsize         =     float(input('labelsize:'))
        
        self._c_cmap = col.LinearSegmentedColormap.from_list('olive',[self._startcolor,self._midcolor,self._endcolor], N = 256, gamma = 1.0)
        cm.register_cmap(cmap = self._c_cmap)
        #font = self._font
        plt.rc('font', **self._font)