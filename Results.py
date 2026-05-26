"""
=============================================================================
Results.py
Author: L.E. van der Hammen
Date: May 26
Description: Processes raw experimental data to calculate maximum axial 
             puncture force, performs a Two-Way ANOVA, and generates a 
             5-panel scientific dashboard for results interpretation.
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
import io

# ==========================================
# 1. RAW DATA INPUT
# ==========================================
# Input your experimental run data here. Ensure the formatting matches exactly.
csv_data = """Run #,Needle Type,Insertion Angle,Number of Nut
1,Blunt,0,2
2,Sharp,30,2
3,Blunt,30,6
4,Sharp,0,0
5,Sharp,30,1
6,Sharp,30,2
7,Blunt,30,4
8,Blunt,0,1
9,Blunt,0,1
10,Sharp,0,0
11,Sharp,30,2
12,Blunt,0,0
13,Sharp,30,1
14,Blunt,30,2
15,Blunt,0,2
16,Sharp,0,0
17,Blunt,30,1
18,Sharp,0,0
19,Sharp,0,0
20,Blunt,30,4"""

# Load the raw string data into a Pandas DataFrame for easier manipulation
df = pd.read_csv(io.StringIO(csv_data))

# ==========================================
# 2. CONSTANTS & FORCE CALCULATION
# ==========================================
M_BASIS = 12.97       # Mass of the base plate [grams]
M_MOER = 4.54         # Mass of a single nut [grams]
G = 9.81              # Gravitational acceleration [m/s^2]

# Calculate the maximum axial force in Newtons.
# Formula: F_axial = (Total Mass in kg) * g * cos(angle)
df['Force_N'] = ((M_BASIS + (df['Number of Nut'] * M_MOER)) / 1000.0) * G * np.cos(np.radians(df['Insertion Angle']))

# Explicitly cast independent variables as categorical data for the ANOVA model
df['Needle Type'] = df['Needle Type'].astype('category')
df['Insertion Angle'] = df['Insertion Angle'].astype('category')

# ==========================================
# 3. STATISTICAL ANALYSIS (ANOVA)
# ==========================================
# Fit an Ordinary Least Squares (OLS) model including the interaction term (*)
model = ols('Force_N ~ C(Q("Needle Type")) * C(Q("Insertion Angle"))', data=df).fit()

# Generate a Type II ANOVA table (standard for balanced factorial designs)
anova_table = sm.stats.anova_lm(model, typ=2)

# Calculate Partial Eta Squared (Effect Size) to determine physical relevance
anova_table['Partial_Eta_Sq'] = anova_table['sum_sq'] / (anova_table['sum_sq'] + anova_table.loc['Residual', 'sum_sq'])
anova_table.loc['Residual', 'Partial_Eta_Sq'] = np.nan

# Extract model residuals to check ANOVA assumptions (homoscedasticity & drift)
df['Residuals'] = model.resid

# ==========================================
# 4. TERMINAL OUTPUTS
# ==========================================
print("="*85)
print(f"{'ADVANCED TWO-WAY ANOVA REPORT':^85}")
print("="*85)
print(anova_table.round(4))
print("="*85)

print("\n" + "="*50)
print(f"{'CHRONOLOGICAL DATA TABLE (TIME VS. FORCE)':^50}")
print("="*50)

# Create a clean display table showing chronological test order
table_df = df[['Run #', 'Needle Type', 'Insertion Angle', 'Force_N']].copy()
table_df['Force_N'] = table_df['Force_N'].round(4)
table_df.rename(columns={'Run #': 'Time (Run Order)', 'Force_N': 'Force [N]'}, inplace=True)
print(table_df.to_string(index=False))
print("="*50)

# ==========================================
# 5. SCIENTIFIC PLOTTING (MASTER DASHBOARD)
# ==========================================
# Apply a clean, academic visual theme
sns.set_theme(style="ticks", context="paper")

# Initialize a 2x3 grid for the dashboard
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# --- Plot A: Raw Data (Swarmplot) ---
# Shows every single data point without overlapping, revealing data clustering
sns.swarmplot(x='Needle Type', y='Force_N', hue='Insertion Angle', data=df, ax=axes[0, 0], dodge=True, palette="Set1", size=8, alpha=0.8)
axes[0, 0].set_title('A. Raw Puncture Force Data', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Maximum Axial Force [N]')
axes[0, 0].set_xlabel('Needle Tip Type')
axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
axes[0, 0].legend(title='Angle [°]', loc='upper left')

# --- Plot B: Boxplot (Distribution & Scale) ---
# Visualizes the median, interquartile ranges, and potential outliers
sns.boxplot(x='Needle Type', y='Force_N', hue='Insertion Angle', data=df, ax=axes[0, 1], palette="Set2", width=0.6, boxprops=dict(alpha=0.8))
axes[0, 1].set_title('B. Distribution & Outliers', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Maximum Axial Force [N]')
axes[0, 1].set_xlabel('Needle Tip Type')
axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
axes[0, 1].legend(title='Angle [°]', loc='upper left')

# --- Plot C: Interaction Effects Plot ---
# Used to determine if the effect of one factor depends on the other (crossing lines = interaction)
sns.pointplot(x='Insertion Angle', y='Force_N', hue='Needle Type', data=df, ax=axes[0, 2], markers=['o', 's'], linestyles=['-', '--'], capsize=0.1, palette="Dark2", err_kws={'linewidth': 1.5})
axes[0, 2].set_title('C. Interaction Effects Plot', fontsize=12, fontweight='bold')
axes[0, 2].set_ylabel('Mean Force [N] ± 95% CI')
axes[0, 2].set_xlabel('Insertion Angle [Degrees]')
axes[0, 2].grid(axis='y', linestyle='--', alpha=0.7)

# --- Plot D: Force Timeline (Systematic Drift Check) ---
# Plots raw force against the run order to detect experimental degradation over time
sns.lineplot(x='Run #', y='Force_N', data=df, ax=axes[1, 0], color='grey', linewidth=1, linestyle='--')
sns.scatterplot(x='Run #', y='Force_N', hue='Needle Type', style='Insertion Angle', data=df, ax=axes[1, 0], s=100, palette='Set1')
axes[1, 0].set_title('D. Force Timeline (Raw Data)', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Maximum Axial Force [N]')
axes[1, 0].set_xlabel('Chronological Timeline (Run Order)')
axes[1, 0].set_xticks(range(1, 21))
axes[1, 0].grid(True, linestyle='--', alpha=0.7)
axes[1, 0].legend(title='Needle / Angle', fontsize=9)

# --- Plot E: Residuals vs. Run Order (ANOVA Drift Check) ---
# Plots statistical error against time. A flat red trendline means randomization was successful.
sns.regplot(x='Run #', y='Residuals', data=df, ax=axes[1, 1], scatter_kws={'s':50}, line_kws={'color':'red', 'linestyle':'--'})
axes[1, 1].set_title('E. Residuals vs. Run Order', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('ANOVA Residuals [N]')
axes[1, 1].set_xlabel('Chronological Timeline (Run Order)')
axes[1, 1].set_xticks(range(1, 21))
axes[1, 1].axhline(0, color='black', linewidth=1)
axes[1, 1].grid(True, linestyle='--', alpha=0.7)

# --- Clean up dashboard layout ---
# Remove the empty 6th subplot for aesthetic purposes
fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.show()

# ==========================================
# 6. ASSUMPTION WARNINGS (REPORT REMINDERS)
# ==========================================
print("\n[CRITICAL REVIEW POINTS FOR DISCUSSION]")
print("- ZERO VARIANCE WARNING: The 'Sharp, 0 degree' group resulted in exactly 0.0 variance. This severely violates ANOVA assumptions of homoscedasticity.")
print("- RELEVANCE CHECK: Review the Effect Size (Partial Eta Sq). Discuss if differences are physically relevant for actual surgical needle insertion.")
print("- SYSTEMATIC DRIFT: Inspect Plots D and E. If the red trendline in E significantly deviates from zero, there was an uncontrolled nuisance variable despite randomization.")