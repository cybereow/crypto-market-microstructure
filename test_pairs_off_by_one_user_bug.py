import pandas as pd
import numpy as np

zscores = [1, 5, 3] # Normal -> StopOut -> Should Enter immediately since 3 > 2
z_exit_threshold = 0.5
z_entry_threshold = 2.0
z_stop_loss = 4.0

current_pos = 0
stopped_out = False
unshifted_pos = []

for i, z in enumerate(zscores):
    if stopped_out:
        if abs(z) < z_exit_threshold:
            stopped_out = False
        # USER BUG IS HERE:
        current_pos = 0
    else:
        if abs(z) > z_stop_loss:
            current_pos = 0
            stopped_out = True
        elif z > z_entry_threshold:
            current_pos = -1
        elif z < -z_entry_threshold:
            current_pos = 1
        elif abs(z) < z_exit_threshold:
            current_pos = 0
    unshifted_pos.append(current_pos)

print("Input Z-Scores:")
print(zscores)
print("Unshifted Positions:")
print(unshifted_pos)
