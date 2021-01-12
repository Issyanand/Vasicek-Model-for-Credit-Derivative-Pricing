# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 13:40:18 2020

"""
import numpy as np
import scipy.integrate as sciIntegr
from scipy import integrate
import matplotlib.pyplot as plt
from scipy import optimize as opt
import scipy.stats as stat
import pandas as pd

class Bond(object):
    def __init__(self,params, r0):
        self.mu=params[0]
        self.sig = params[1]
        self.kappa =params[2]
        self.r0=r0

    def A(self,t,T):
        pass

    def B(self,t,T):
        pass

    def Exact_ZCB_Price(self,t,T):
        pass

    def Sim_Euler(self, M,I,T):
        pass

    def SpotRate(self, t, T):
        price = self.Exact_ZCB_Price(t, T)
        time  = T - t
        return (-np.log(price)/time)

    def StochasticPrice(self,M, I, tau):
        # VectorRates is a two dimensional array:
        # with simulated rates in columns and timesteps rates in rows

        # we do not need VectorRates and VectorTime at the beginning of the simulation as it is r0
        v_times,v_rates = self.Sim_Euler(M, I, tau)

        No_Sim = v_rates.shape[1]

        price  = np.zeros(No_Sim)
        for i in range(No_Sim):
            Rates    = v_rates[:,i].T
            price[i] = np.exp(-(sciIntegr.simps(Rates , v_times)))

        RangeUp_Down   = np.sqrt(np.var(price))*1.96 / np.sqrt(No_Sim)
        Mean = np.mean(price)

        return Mean,  Mean + RangeUp_Down, Mean - RangeUp_Down


class Vasicek_Bond(Bond):

    def __init__(self,params,r0):
        Bond.__init__(self,params,r0)

    def B(self, t, T):
        return (1 - np.exp(-self.mu*(T-t))) / self.mu

    def A(self, t, T):
        """This is the formula from Github"""
        return ((self.mu-(self.sig**2)/(2*(self.kappa**2))) *(self.B(t, T)-(T-t)) \
                   -  (self.sig**2)/(4*self.kappa)*(self.B(t, T)**2))


    def Exact_ZCB_Price(self,t,T):
        B = self.B(t,T)
        A = self.A(t,T)
        return np.exp(A-self.r0*B)

    def Sim_Euler(self, M, num_trials,tau):

        dt = tau/M

        xh = np.zeros((M + 1, num_trials))
        rates = np.zeros((M + 1, num_trials))
        times = np.linspace(0, tau, M + 1)
        xh[0]     = self.r0

        for i in range(1, M + 1):
            xh[i] = (xh[i - 1] +
                  self.kappa * (self.mu - xh[i - 1]) * dt + \
                  self.sig * np.sqrt(dt) * np.random.standard_normal(num_trials))
        rates = xh
        return (times, rates)

    def ExpectedRate(self,t, T):
        result = self.r0 * np.exp(-self.kappa*(T-t)) + self.mu*(1-np.exp(-self.kappa*(T-t)))
        return result

class Vasicek_Defaultable_Bond(Vasicek_Bond):

    def __init__(self,params_sr,params_ir,r0,l0,rho):
        Bond.__init__(self,params_sr,r0)
        self.mu_ir = params_ir[0]
        self.sig_ir= params_ir[1]
        self.kappa_ir = params_ir[2]
        self.l0 = 0
        self.rho = rho


    def Defaultable_Exact_ZCB_Price(self,t,T):
        """
        Need analytical price for Defaultable Bond
        """
        default_free = self.Exact_ZCB_Price(t,T)
        p_def =self.prob_default(t,T)
        return default_free*p_def

    ###HELPER FUNCTIONS#############################

    def prob_default(self,t,T):
        #calculates and returns the cumulative default probability
        k_tilde = self.kappa_ir - self.rho*self.sig_ir*self.sig*self.B_ir(t,T)
        A_ = self.A_ir(t,T,self.mu_ir,self.sig_ir,k_tilde)
        B_ = self.B_ir(t,T)
        return np.exp(A_ - B_*self.l0)


    def B_ir(self, t, T):
        #Returns the B expression for the intensity process
        return (1 - np.exp(-self.mu_ir*(T-t))) / self.mu_ir

    def A_ir(self, t, T,mu,sig,k):
        #Returns the A expression for the intensity process
        return ((mu-(sig**2)/(2*(k**2))) *(self.B_ir(t, T)-(T-t)) \
                   -  (self.sig**2)/(4*k)*(self.B_ir(t, T)**2))

    def StochasticPrice(self,M, I, tau):
        # VectorRates is a two dimensional array:
        # with simulated rates in columns and timesteps rates in rows

        # we do not need VectorRates and VectorTime at the beginning of the simulation as it is r0
        v_times,s_rates,i_rates = self.Sim_Euler(M, I, tau)

        No_Sim = s_rates.shape[1]

        def_free_price  = np.zeros(No_Sim)
        def_prob = np.zeros(No_Sim)
        price  = np.zeros(No_Sim)
        for i in range(No_Sim):
            SRates    = s_rates[:,i].T
            IRates =i_rates[:,i].T
            def_free_price[i] = np.exp(-(sciIntegr.simps(SRates , v_times)))
            def_prob[i] = np.exp(-(sciIntegr.simps(IRates , v_times)))
            price[i] = def_free_price[i] * def_prob[i]
        RangeUp_Down   = np.sqrt(np.var(price))*1.96 / np.sqrt(No_Sim)
        Mean = np.mean(price)

        return Mean,  Mean + RangeUp_Down, Mean - RangeUp_Down

    def Sim_Euler(self,M,num_trials,tau):
        """
        2D Simulation for short rate and default process
        """
        #s_rates is short rate process
        #i_rates is def intensity process

        dt = tau/M

        xh = np.zeros((M + 1, num_trials))
        yh = np.zeros((M + 1, num_trials))

        s_rates = np.zeros((M + 1, num_trials))
        i_rates = np.zeros((M + 1, num_trials))
        times = np.linspace(0, tau, M + 1)

        xh[0] = self.r0
        yh[0] = self.l0
        for i in range(1, M + 1):
            zed1 = np.random.normal()
            zed2 = np.random.normal()
            Z1 = zed1
            Z2 = self.rho*zed1 + np.sqrt(1-self.rho**2)*zed2
            xh[i] = (xh[i - 1] +
                  self.kappa * (self.mu - xh[i - 1]) * dt + \
                  self.sig * np.sqrt(dt) * Z1)
            yh[i] = (yh[i - 1] +
                  self.kappa_ir * (self.mu_ir - yh[i - 1]) * dt + \
                  self.sig_ir * np.sqrt(dt) * Z2)
        s_rates = xh
        i_rates = yh
        return (times, s_rates,i_rates)

def calibrate_sr(data,t):

    s_y = data[1:]
    s_x = data[:len(data)-1]

    n =len(data)
    dt = 1/252

    sx = sum(s_x)
    sy = sum(s_y)
    sxx = sum(s_x*s_x)
    syy = sum(s_y*s_y)
    sxy = sum(s_x*s_y)

    a = (n*sxy -sx*sy)/(n*sxx - sx*sx)
    b = (sy - a*sx)/n

    sde = np.sqrt((n*syy - sy*sy - a*(n*sxy - sx*sy))/(n*(n-2)))
    res_lambdy = -np.log(a)/dt
    res_mu = b/(1-a)
    res_sigma = sde*np.sqrt(-2*np.log(a)/(dt*(1-a**2)))



    return [res_mu,res_sigma, res_lambdy]


#Caibration to Treasury Yields
# -> Calibration error shrinks when using diff r0 for each maturity
#Previously used r0=.02, but this gave worse results
rates_filename = 'Treasury Yield.xlsx'
rates = pd.read_excel(rates_filename)
rates.index = rates['Date']

rates = rates.iloc[:,1:]
rates= rates/100
taus = [1/12,2/12,3/12,6/12,1,2,3,5,7,10,20,30]

#result arrays for calibrated prices and mkt prices
res_params = np.zeros((len(taus),3))
prices= np.zeros(len(taus))
mkt_prices = np.zeros(len(taus))
stoch_prices = np.zeros(len(taus))

#result arrays for defaultable results
p_def = np.zeros(len(taus))
def_prices = np.zeros(len(taus))
stoch_def_prices = np.zeros(len(taus))

#Assume the parameters of some intensity process
params_ir = [.03,.05,2.0]
l0=.03
rho = -.5

#for each maturity in the dataset
for i in range(len(taus)):
    #first calibrate the short rate and price the def-free bonds
    t = taus[i]
    res_params[i] = calibrate_sr(rates.iloc[:,i].values,t)
    r0 = rates.iloc[-1,i]
    vasi = Vasicek_Bond(res_params[i],r0)
    prices[i] =vasi.Exact_ZCB_Price(0,t)
    mkt_prices[i] = np.exp(-r0*t)
    stoch_prices[i] = vasi.StochasticPrice(252,10000,t)[0]

    #next find out defaultable bond prices, using assumed parameters
    vasi_def = Vasicek_Defaultable_Bond(res_params[i],params_ir,r0,l0,rho)
    p_def[i] = vasi_def.prob_default(0,t)
    def_prices[i] = prices[i]*p_def[i]
    stoch_def_prices[i] = vasi_def.StochasticPrice(252,10000,t)[0]

calibration_error = abs(prices-mkt_prices)
stoch_error = abs(stoch_prices-prices)
stoch_def_error = abs(stoch_def_prices - def_prices)

plt.plot(taus,100*calibration_error/mkt_prices)
plt.plot(taus,100*stoch_error/prices)
#plt.plot(taus,100*stoch_def_error/def_prices)
