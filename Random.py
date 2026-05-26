"""
=============================================================================
File: Random.py
Author: L.E. van der Hammen
Date: May 24, 2026
Description: Generates a fully randomized, balanced run table (n=5) for 
             a 2x2 factorial design (Needle Tip x Insertion Angle).
=============================================================================
"""

import random

# Set a fixed random seed. This ensures that every time you run this script, 
# you get the exact same "random" sequence. This is crucial for scientific 
# reproducibility and verifying your run order later.
random.seed(42)

# Define the 4 Experimental Conditions (ECs) for the 2x2 factorial design.
# Factor A: Needle Tip (Sharp, Blunt)
# Factor B: Insertion Angle (0°, 30°)
conditions = [
    ('Sharp', '0°'),
    ('Sharp', '30°'),
    ('Blunt', '0°'),
    ('Blunt', '30°')
]

# Repeat each experimental condition exactly 5 times. 
# This creates a balanced design with a total of 20 runs (n=5 per group).
run_list = conditions * 5

# Completely shuffle the run order (Randomization). 
# This neutralizes nuisance variables (e.g., gelatin degradation over time).
random.shuffle(run_list)

# ==========================================
# TERMINAL OUTPUT: RUN TABLE
# ==========================================
print(f"{'Run #':<8}{'Needle Tip':<15}{'Insertion Angle':<15}")
print("-" * 38)

# Iterate through the randomized list and print each run with its corresponding number
for index, condition in enumerate(run_list, start=1):
    needle_tip = condition[0]
    angle = condition[1]
    
    print(f"{index:<8}{needle_tip:<15}{angle:<15}")