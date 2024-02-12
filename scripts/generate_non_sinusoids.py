# Authors: Dylan Jones
# Date: 1/21/24
# Purpose: Generate a set of arbitrary non-sinusoidal waveforms so that models can
#          converge on Sin(t).

import matplotlib.pyplot as plt
import os
import sys
# sys.path.insert(0, "../modules/")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../modules')))
from waveforms import *

# Change the working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Period length of waveforms
PERIOD = 100

# If you want to wipe all the current plots in the plots directory before
# making more
WIPE_PLOTS = True

if (WIPE_PLOTS):
    for plot in os.listdir("plots"):
        plot_path = os.path.join("plots", plot)
        os.remove(plot_path)

# Collection of waveforms to train with
squares = []
trapezoids = []
triangles = []

# Add square waves
for duty in range(10, 90, 2):
    duty_percentage = duty / 100
    squares.append(Square_wave(duty_cyle=duty_percentage, dc_bias=4))

# Add trapezoid waves
for top_time in range(10, 50, 2):
    for top_width in range(5, 25, 5):
        for bot_time in range((top_time + top_width + 10), PERIOD):
            trapezoids.append(Trapezoid_wave(t_reach_top=top_time, 
                                                      top_width=top_width, 
                                                      t_reach_bot=bot_time,
                                                      dc_bias=4))

# Add triangle waves
for top_time in range(10, 50, 2):
    for bot_time in range(top_time + 10, PERIOD):
        triangles.append(Trapezoid_wave(t_reach_top=top_time, 
                                                      top_width=0, 
                                                      t_reach_bot=bot_time,
                                                      dc_bias=4))
        

# Add all waveforms together
waveform_collection = squares + trapezoids + triangles

print("Number of waveforms:", len(waveform_collection))
for i in range(1, len(waveform_collection), 1):
    num_string = f"waveform{i}"
    plot_name = num_string.zfill(4)
    file_path = os.path.join("plots", plot_name)
    plt.plot(waveform_collection[i].wave)
    plt.ylim(bottom=0, top=8)
    plt.xlabel("Time Units (1 Fundamental Period)")
    plt.ylabel("Voltage Units (Arb)")
    plt.savefig(file_path)
    plt.clf()