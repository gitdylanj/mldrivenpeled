import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

def pieceWise(arr: np.ndarray, domains, funcs) -> np.ndarray:
    '''
    Takes an array and a set of boolean domains and functions
    and returns the piecewise output of the functions for the domains
    '''
    output = np.empty(len(arr))
    for domain, func in zip(domains, funcs):
        output[domain] = func(arr[domain])
    return output

class Waveform:
    '''
    Represents a base waveform for a period window with a DC offset
    '''
    def __init__(self, v_peak=1.0, length=100, wave_function=lambda x: x, dc_bias=1):
        '''
        Builds a waveform with default length and DC offset of 1
        '''
        self.array = np.arange(0, length)
        self.period = len(self.array)
        self.dc_bias = dc_bias
        self.v_peak = v_peak
        self.__function = wave_function # Must be amplitude 1 to work well with v_peak
        self.wave = self.__create_wave()

    def __create_wave(self):
        return self.v_peak * self.__function(self.array) + self.dc_bias

class Sinusoid(Waveform):
    '''Represents a simple sine wave'''
    def __init__(self, v_peak=1, dc_bias=1):
        super().__init__(v_peak, dc_bias=dc_bias, wave_function=self.__sinusoid)

    def __sinusoid(self, t):
        return np.sin((2 * np.pi / self.period) * t)  
    
class Square_wave(Waveform):
    '''
    Represents a square wave where the duty cycle specifies what percentage
    of the period the voltage is ON
    '''
    def __init__(self, duty_cyle=0.5, v_peak=1, dc_bias=1):
        self.duty_cyle = duty_cyle
        self.v_peak = v_peak
        self.adjusted_v_peak = self.v_peak / 2.0
        self.dc_bias = dc_bias
        super().__init__(v_peak=self.adjusted_v_peak, dc_bias=dc_bias, wave_function=self.__square_func)
        
    def __square_func(self, t):
        adjusted_period = 2 * self.duty_cyle * self.period
        return np.where(t < adjusted_period,
                        np.sign(np.sin((2 * np.pi / adjusted_period) * t)),
                        (-1))

class Trapezoid_wave(Waveform):
    '''
    Builds a trapezoid (or a triangle wave if top_width is 0) by 
    taking in a specified t value for reaching the top of the trapezoid,
    the width of the top of the shape, and the t value at which it reaches
    the bottom
    '''
    def __init__(self, t_reach_top, t_reach_bot, top_width=0, duty_cyle=0.5, v_peak=1, dc_bias=1):
        self.duty_cyle = duty_cyle
        self.t_reach_top = t_reach_top
        self.t_reach_bot = t_reach_bot
        self.top_width = top_width
        super().__init__(v_peak, dc_bias=dc_bias, wave_function=self.__trap_func)

    def __trap_func(self, t):
        if (self.top_width + self.t_reach_top >= self.t_reach_bot):
            raise ValueError("Invalid point specified for t_reach_bot")
        domains = [
            (t < self.t_reach_top),
            (np.logical_and(t >= self.t_reach_top, t < (self.t_reach_top + self.top_width))),
            (np.logical_and(t >= (self.t_reach_top + self.top_width), t < self.t_reach_bot)),
            (t >= self.t_reach_bot)
            ]
        piece_funcs = [
            lambda t: (1.0 / self.t_reach_top) * t,
            lambda t: 1.0,
            lambda t: (1 / (self.t_reach_bot - (self.t_reach_top + self.top_width))) * (self.t_reach_bot - t),
            lambda t: 0
                       ]
        return pieceWise(t, domains, piece_funcs)