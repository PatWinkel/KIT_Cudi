# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 10:52:09 2016

@author: Patrick
"""

import numpy as np
import scipy.special as ss
import scipy.constants as sc
import matplotlib.pyplot as plt

phi_0 = sc.physical_constants['magn. flux quantum'][0]
c = sc.physical_constants['speed of light in vacuum'][0]   
h = sc.h
hbar = sc.h/(2*np.pi)
e = sc.e 

class cpw_resonator(object):
    
    
    def __init__(self,w,g,l, Cc):
        self._name = 'resonator'   
        
        self._w = w      #center line width
        self._g = g      #gap between center line & ground planes
        self._h1 = 350e-6           #substrate height
        self._h3 = 3000e-6           #distance to samplebox above chip
        self._h4 = 3000e-6           #distance to samplebox under chip
        self._e1 = 11.9           #dielectric constant substrate
        self._l = l             #resonator length
        self._Cc = Cc    #coupling capacitor
        self._theta0 = np.pi     #offset angle at resonance frequency
        
        self._Z0 = 50.             
        self._tan = 8.5e-6      #loss tangent
        self._Qint = 1/self._tan

 
        self.prep_values()
        
        self._font              =    {'weight' : 'normal', 'size' : 20}
        self._labelsize         =    27
        plt.rc('font', **self._font)
        
        
    def set_length(self,length):
        '''resonator length'''
        self._l = length
        
    def set_couplingc(self, Cc):
        '''coupling capacitor'''
        self._Cc = Cc
    
    def set_Z0(self, Z0):
        '''cpw impedance'''
        self._Z0 = Z0
    
    def set_height(self,h1,h3,h4):
        '''substrate height'''
        self._h1 = h1
        self._h3 = h3
        self._h4 = h4
        
    def set_diel(self, e1):
        '''dielectric constant (substrate)'''
        self._e1 = e1
    
    def set_tan(self,tan):
        '''loss tangent (substrate)'''
        self._tan = tan
        
    
        
    def alpha(self):
        return np.pi*np.sqrt(self._eff)*self._fres*self._tan/c
        
    def beta(self):
        return  np.sqrt(self._L*self._C)
        
    def gamma(self, f):
        return 1.j*(2.*np.pi*f)*self._beta + self._alpha*self._l
        
    
    def prep_values(self):
        self.conformal(w = None, g = None)
        self.fres()
        self._alpha = self.alpha()
        self.dist()
        self.lumped()
        self.fresc()
        self.Q_ext(self._wresc, self._Cc)
        self.Q_load()
        self._beta = self.beta()
        
        self.set_freq()
        
        self.S11(self._freq)
        self.S11_decay(self._freq)


        
    def set_freq(self):
        self._fmin = self._fresc - 100e6
        self._fmax = self._fresc + 100e6
        
        self._freq = np.arange(self._fmin, self._fmax,1e4)
        
        self._df = np.arange(-1,1,0.001)
        
    def get_values(self):
        print 'Ll = {} H/m'.format(self._Ll)
        print 'Cl = {} F/m'.format(self._Cl)
        print 'distributed:'
        print 'L = {} H'.format(self._L)
        print 'C = {} F'.format(self._C)
        print 'R = {} Ohm'.format(self._R)
        print 'lumped element:'
        print 'Lr = {} H'.format(self._Lr)
        print 'Cr = {} F'.format(self._Cr)
        print 'Rr = {} Ohm'.format(self._Rr)
        print
        print 'Q_int = {} '.format(self._Qint)
        print 'Q_ext = {} '.format(self._Qext)
        print 'Q_load = {} '.format(self._Qload)
        print 
        print 'f_res = {} GHz'.format(self._fres)
        print 'f_resc = {} GHz '.format(self._fresc)
        
    
    def k_tanh(self,h,w,g):
        return np.tanh(np.pi*w/(4.*h))/np.tanh(np.pi*(w+2.*g)/(4.*h))

    def k_sinh(self,h,w,g):
        return np.sinh(np.pi*w/(4.*h))/np.sinh(np.pi*(w+2.*g)/(4.*h))
        
        
    def conformal(self, w = None, g = None):                            #calculates the inductance and capacitance per unit length
        internal = False
        if w == None:
            internal = True
            w = self._w
            g = self._g
        k1 = self.k_sinh(self._h1,w,g)
        k3 = self.k_tanh(self._h3,w,g)
        k4 = self.k_tanh(self._h4,w,g)
        K1 = ss.ellipk(k1)
        K12 = ss.ellipk((1-k1**2.)**0.5)
        K3 = ss.ellipk(k3)
        K32 = ss.ellipk((1-k3**2.)**0.5)
        K4 = ss.ellipk(k4)
        K42 = ss.ellipk((1-k4**2.)**0.5)
        k0 = w/(w+2.*g)
        k01 = (1-k0**2.)**0.5
        K0 = ss.ellipk(k0)
        K01 = ss.ellipk(k01)
        q1 = (K1/K12)*(K3/K32 + K4/K42)**(-1)
        if  internal:    
            self._eff = 1. + (self._e1-1)*q1
            self._Cl = 4*sc.epsilon_0*self._eff*(K0/K01)
            self._Ll = sc.mu_0/4*(K01/K0)
        else: 
            return 1. + (self._e1-1)*q1, 4*sc.epsilon_0*self._eff*(K0/K01), sc.mu_0/4*(K01/K0)

    
        
     
    def imp_Al_CPW(self):     
        '''impedance Z for an Al CPW'''                 
        self._Z = (self._Ll/self._Cl)**0.5
        return self._Z
 

    def dist(self):
        '''distributed resonator properties'''
        self._L = self._Ll*self._l
        self._C = self._Cl*self._l
        self._R = 2*self._alpha*self._Z0
    
    def lumped(self):
        '''lumped element equivalent'''
        self._Lr = 8*self._Ll*self._l/np.pi**2
        self._Cr = self._Cl*self._l/2
        self._Rr = 2*self._Z0/(self._alpha*self._l)

    def fres(self):
        '''uncoupled resonance frequency (bare resonator)'''
        self._fres = c/(4.*self._l*(self._eff)**0.5)
        self._wres = 2*np.pi*self._fres
        
    
    
    def fresc(self):
        '''coupled/loaded resonance frequency'''
        self._C_star = self._Cc/(1+(2*np.pi*self._fres*self._Cc*self._Z0)**2)
        self._fresc = 1/(2*np.pi*np.sqrt(self._Lr*(self._Cr+self._C_star)))
        self._wresc = 2*np.pi*self._fresc
        
    def Q_ext(self, wresc, Cc, own = True):
        '''external quality factor'''
        self._Rstar = (1+(wresc*Cc*self._Z0)**2.)/(wresc**2.*Cc**2.*self._Z0)
        if own:
            self._Qext = self._wresc*(self._Cr+Cc)*self._Rstar
            self._ke = self._wresc/(2*np.pi*self._Qext)     #external decay rate (due to coupling capacitor)
            self._ki = self._wresc/(2*np.pi*self._Qint)     #internal decay rate due to losses
        else:    
            return wresc*(self._Cr+Cc)*self._Rstar
        
    def Q_load(self):
        '''loaded quality factor'''
        self._Qload = (1/self._Qext + 1/self._Qint)**(-1)
            
        
        
    def R_star(self, Cc, Z0, wresc):
        return (1+(wresc*Cc*Z0)**2.)/(wresc**2.*Cc**2.*Z0)
        
    
    
    def Z_Al_CPW(self,w,g):                      #impedance Z for an Al CPW 
        L = self.conformal(w,g)[2]
        C = self.conformal(w,g)[1]
        Z = (L/C)**0.5
        return Z        
        
        
        
        
        
    '''input impedance: CPW resonator - lumped element model:'''    
        
    def Z_in(self,f):
        '''parallel LRC resonator with coupled capacitively (Cc) to a transmission line'''
        w = 2.*np.pi*f
        return 1./(1.j*w*self._Cc)+1./(1./self._Rr+1./(1.j*w*self._Lr)+1.j*w*self._Cr)
        
    def Z_in_res(self,f):
        '''parallel LRC resonator - no coupling'''
        w = 2.*np.pi*f
        return 1./(1./self._Rr+1./(1.j*w*self._Lr)+1.j*w*self._Cr)




    '''input impedance: lumped element model:'''   
        
    def Z_in_LRC_P(self, f, L, C, R):
        '''parallel LRC resonator - no coupling'''
        w = 2.*np.pi*f
        return  1./(1./R+1./(1.j*w*L)+1.j*w*C)      
    
    def Z_in_LRC_C(self, f, L, C, R, Cc):
        '''parallel LRC resonator - coupled'''
        w = 2.*np.pi*f
        return  1./(1.j*w*Cc)+1./(1./R+1./(1.j*w*L)+1.j*w*C)  
        
    def Z_in_Norton(self, f, L, C, R, C_star):
        '''parallel LRC resonator - coupled; Norton transformed'''
        w = 2.*np.pi*f
        return  (1.j*w*C_star+1./R+1./(1.j*w*L)+1.j*w*C)**(-1) 
        
    def Z_in_LRC_PC(self, f, L, C, R, R_star, C_star):
        '''parallel LRC resonator - coupled'''
        w = 2.*np.pi*f
        return  (1.j*w*C_star+1/R_star + 1./R+1./(1.j*w*L)+1.j*w*C)**(-1)    
        
    def Z_in_LRC_S(self, f, L, C, R):
        '''seriel LRC resonator - no coupling'''
        w = 2.*np.pi*f
        return  R + 1.j*w*L - 1.j*1./(w*C)       
        
    def S11_LRC(self, f, Z_in, Z0):
        '''S11 matrix element for input impedance Z_in & line impedance Z0'''
        return (Z_in-Z0)/(Z0+Z_in)
        
        
        
        
        
        
        
        
    def mag(self, S11):
        '''magnitude S11'''
        return np.sqrt(S11.imag**2.+S11.real**2.)
    
    def arg(self, S11):
        '''phase S11'''
        return np.unwrap(np.angle(S11))+self._theta0
    
    def S11(self,f):
        '''S11 matrix entry without any assumptions or approximation'''
        self._S11 =  (self.Z_in(f)-self._Z0)/(self.Z_in(f)+self._Z0)
        self._mag = self.mag(self._S11)
        self._arg = self.arg(self._S11)
    
    '''
    S11 matrix element close to the resonance freq in terms of the internal and 
    external decay rate ki and ke, respectively
    '''
    
    def S11_decay(self,f):
        df = f - self._fresc
        self._S11_2 = (df**2. + 0.25*(self._ki**2.-self._ke**2.)+1.j*self._ke*df)/(df**2.+0.25*(self._ki+self._ke)**2.) 
        self._mag_dec = self.mag(self._S11_2)
        self._arg_dec = self.arg(self._S11_2)

    def S11_ref(self,f):
        return ( 2.*self._Qload/self._Qext - 1. + 2.j*self._Qload*(self._fresc-f)/self._fresc ) / ( 1. - 2.j*self._Qload*(self._fresc-f)/self._fresc )  
        
    def S11_Q(self,f, Ql, Qext, fresc):
        return ( 2.*Ql/Qext - 1. + 2.j*Ql*(fresc-f)/self._fresc ) / ( 1. - 2.j*Ql*(fresc-f)/fresc )  
    '''
    S11 matrix element taken from Sebastian Probst's circle fit routine
    '''
    def _S11_seb(self,f):
        '''
        use either frequency or angular frequency units
        for all quantities
        k_l=k_c+k_i: total (loaded) coupling rate
        k_c: coupling rate
        k_i: internal loss rate
        '''
        return ((self._ke-self._ki)+2.j*(f-self._fresc))/((self._ke+self._ki)-2.j*(f-self._fresc))
        
        
        
    '''introducing a SQUID termination'''    
        
    def set_Ic(self, Ic):
        '''ciritcal current single junction'''
        self._Ic = Ic
    
    def EJ(self, Ic):
        '''Josephson energy'''
        return phi_0*Ic/(2*np.pi)
    def LJ(self, Ic):
        '''Josephson inductance
        insert Ic = Isum for DC-SQUIDs
        '''
        return phi_0/(2.*np.pi*Ic)
    
    def EL_cav(self):
        '''inductive energy (cavity)'''
        return (hbar/(2.*e))**2./(self._Ll*self._l)
    
    def EC(self, CJ):
        '''qarging energy'''
        return (2*e)**2./(2.*CJ)
        
    def wJ(self, Ic, CJ):
        '''JJ plasma frequency'''
        return np.sqrt(2*self.EC(CJ)*self.EJ(Ic))/hbar
        
        
    def Ic_asym(self,Ic1, Ic2,F):      
        '''effective critical current for an asymmetric DC-SQUID
        F: applied flux
        d: junction asymmetry'''
        self._Ic0_asym = Ic1+Ic2
        d = np.absolute(Ic1-Ic2)/self._Ic0_asym
        return np.abs(np.cos(np.pi*F))*self._Ic0_asym*(1+d**2.*(np.tan(np.pi*F))**2.)**0.5
        
    def L_SQUID_asym(self,Ic1,Ic2,F):
        '''SQUID inductance (asymmetric junctions)'''
        return phi_0/(2.*np.pi*self.Ic_asym(Ic1,Ic2,F))
                
        
    def Ic_sym(self,Ic,F):
        '''effective critical current for a symmetric DC-SQUID'''
        self._Ic0_sym = 2*Ic #F = 0
        return 2*Ic*np.abs(np.cos(np.pi*F))
    
    def L_SQUID_sym(self,Ic,F):
        '''SQUID inductance (symmetric junctions)'''
        return phi_0/(2.*np.pi*self.Ic_sym(Ic,F))


    '''WUSTMANN paper theoy'''
        
    def f_wustmann(self,Ic,F):
        '''resonance frequency of SQUID-terminated quaterwave resonator (WUSTMANN paper)
        F is the flux quanta normalized applied magnetic flux
        Ic is the effective critical current of the SQUID'''
        self._LJ = phi_0/(2*np.pi*Ic/2.)     #jj inductance zero flux
        self._gamma = self._LJ/(2*self._Ll*self._l*np.cos(F*np.pi))     #participation ratio
        return (1-self._gamma)/(4*self._l*np.sqrt(self._Ll*self._Cl))
     
    def gamma_wust(self,Ic, F):
         '''flux dependent participation ratio; Ic: critical current JJ'''
         return self.EL_cav()/(2.*self.EJ(Ic)*np.cos(F*np.pi))
    
    def knd(self, Ic, F, n):
        '''mode spectrum k_n * d (d: resonator length) for small participation ratios gamma'''
        k0d = np.pi/2.*(1.-self.gamma_wust(Ic,F))
        return k0d+np.pi*n
        
     
    def Mn(self,CJ, Ic, F, n):
        ''' 'masses' of the mode oscillators'''
        return 1.+ np.sin(2.*self.knd(Ic,F,n))/(2.*self.knd(Ic,F,n))+4.*CJ/(self._Cl*self._l)*np.cos(self.knd(Ic, F, n))**2.
        
        
     
    def alpha_SQUID(self,w0, CJ, Ic, F):
        '''SQUID non-linearity due to Josephson junction - much larger than cavity non-linearity for gamma << 1
        w0 is the fundamental cavity mode'''
        return hbar*w0**2./(2*self.gamma_wust(Ic,F)*self.EL_cav())*(np.cos(self.knd(Ic, F, 0.))/(self.Mn(CJ, Ic, F, 0.)*(self.knd(Ic, F, 0.))**2.))**2.
        
    def alpha_resonator(self,w0, gamma):
        '''cavity non-linearity due to Josephson junction - only valid for gamma << 1; 
        --> frequency shift per photon inside the cavity (attention: omega; circular frequency)
        w0 is the fundamental cavity mode'''
        return gamma**3.*hbar*w0**2./(2.*self.EL_cav())
        
        
    '''WALLRAFF paper theory'''    
        
    def fres_Wallraff(self,Ic1, Ic2,F,N): 
        '''Ic: (effective) critical current'''
        Lj = N*self.L_SQUID_asym(Ic1, Ic2, F)     #Josephson inductance
        kd = 0.5*np.pi/((1+Lj/(self._Ll*self._l)))             #mode spectrum
        L = ((kd)**2/(2*self._Ll*self._l)*(1+np.sin(2*kd)/(2*kd)))**(-1)    #effective lumped element inductance
        C = 0.5*(self._Cl*self._l)*(1+np.sin(2*kd)/(2*kd))                 #effective lumped element capacitance
        return 1/(2*np.pi*np.sqrt(L*(C+self._Cc)))
        

        
        
        
    def plot_complex(self, S11):
        self._fig, axes = plt.subplots(figsize = (8,8))
        axes.scatter(S11.real,S11.imag, lw = 5.)
        axes.set_xlabel('real')
        axes.set_ylabel('imag')

    def plot_S11(self, mag = True, log = False):
        self._fig, axes = plt.subplots(figsize=(8,4))
        #self.S11(f)
        if mag:
            if log:
                axes.plot(self._freq,20*np.log10(self._mag), 'k', label = r'$|S_{11}|$', color = 'blue', lw = 5.)
                axes.set_ylabel(r'$|S_{11}|^2 \, (\mathrm{dB})$', fontsize = self._labelsize )
            else:
                axes.plot(self._freq,self._mag, 'k', label = r'$|S_{11}|$', color = 'blue', lw = 5.)
                axes.set_ylabel(r'$|S_{11}| \, (\mathrm{a.u.})$', fontsize = self._labelsize )
        else:
            axes.plot(self._freq,self._arg, 'k', label = r'$\arg (S_{11})$', color = 'blue', lw = 5.)
            axes.set_ylabel(r'$ \arg (S_{11}) \, (\mathrm{rad})$', fontsize = self._labelsize )
            axes.set_ylim(-np.pi,np.pi)
        axes.set_xlabel('$f$ [GHz]', fontsize = self._labelsize )
        axes.set_xlim(self._fmin,self._fmax)

            
            
      
    def plot_save(self, add):
        '''function to save plots'''  
        self._fig.savefig(str(self._name+add), dpi = self._dpi)    
        
    def plot(self,f,arr):
        '''function to plot array over f'''
        self._fig, axes = plt.subplots(figsize = (8,4))
        axes.plot(f,arr)
        
        
        
        
    def plot_single(self,f, imp = True, log = False):
        fig, axes = plt.subplots(figsize=(8,4))
        if imp:
            axes.plot(f,self.Z_in(f), 'k', label = '$Z_{in}$', color = 'blue')
            axes.set_ylabel('input impedance $Z_in$ [dB]', fontsize = self._labelsize )
        else:
            axes.plot(f,self.S11(f), 'k', label = '$\tau$', color = 'blue')
            axes.set_ylabel('reflection coefficient $\Gamma$', fontsize = self._labelsize )
        axes.set_xlabel('$f$ [GHz]', fontsize = self._labelsize )
        if log:        
            axes.set_yscale('log')
        
        axes.legend()
        #plt.show()
        