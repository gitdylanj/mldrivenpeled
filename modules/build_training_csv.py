import pandas as pd
import numpy as np
DEFAULT_BIAS = 3
class Testing_file_maker:
    '''
    Creates a batch testing csv of training waveforms given an interable
    collection of waveform arrays. Adds DC bias and waveform frequency 
    information
    '''
    def __init__(self, waves: list[np.ndarray],
                freq_bottom: float,
                freq_top: float,
                num_freq_steps: int=9,
                dc_bias: float = DEFAULT_BIAS,
                bias_bot: None | float = None,
                bias_top: None | float = None,
                bias_step_size: float = 0.2
                ):
        '''
        Takes a set of waveform arrays waves and a minimum and maximum frequency. Each wave is assigned
        a frequency starting from freq_bottom and stepping in equally spaced intervals to freq_top.
        Moreover it is possible to specify a dc bias for all waveforms. If bias_top and bias_bot are given,
        ranges of dc_biases will be generated for each waveform similar to the frequencies.
        '''
        self.waves = waves
        self.freq_bottom = freq_bottom
        self.freq_top = freq_top
        self.num_freq_steps = num_freq_steps

        # Create frequency ranges
        frequencies = [round(freq, 1) for freq in np.arange(freq_bottom, (freq_top), (freq_top - freq_bottom) / num_freq_steps)]
        if (frequencies[-1] != freq_top):
            frequencies.append(freq_top)

        # Create DC offsets
        if ((bias_top is not None) and (bias_bot is not None)):
            offsets = [round(offset, 1) for offset in np.arange(bias_bot, (bias_top), bias_step_size)]
            if (offsets[-1] != bias_top):
                offsets.append(bias_top)
        else:
            offsets = [dc_bias]
        
        # Create dataframe
        num_wave_points = len(waves[0])
        rows = []
        for waveform_array in waves:
            for freq in frequencies:
                for dc_bias in offsets:
                    new_row = {"frequency": freq, "dc_bias": dc_bias}
                    wave_points = {f"v{i}": waveform_array[i] for i in range(num_wave_points)}
                    new_row.update(wave_points)
                    rows.append(new_row)
        df = pd.DataFrame(rows)

        self.training_dataframe = df