
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind


# LOAD DATA

df = pd.read_csv("data/goalkeepers_clean.csv")


# BASIC SUMMARY

print("\n===== Dataset Head =====")
print(df.head())

print("\n===== Summary Statistics =====")
print(df.describe())

# VISUALIZATIONS


# --- Save Percentage Distribution ---
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="save_percentage", hue="type", kde=True)
plt.title("Save Percentage Distribution")
plt.xlabel("Save Percentage")
plt.ylabel("Frequency")
plt.savefig("figures/save_percentage_distribution.png")
plt.close()

# --- Successful Passes Boxplot ---
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="type", y="successful_passes")
plt.title("Successful Passes by Goalkeeper Type")
plt.xlabel("Goalkeeper Type")
plt.ylabel("Successful Passes")
plt.savefig("figures/passing_boxplot.png")
plt.close()

# --- Correlation Heatmap ---
plt.figure(figsize=(10, 6))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("figures/correlation_heatmap.png")
plt.close()


# HYPOTHESIS TEST FUNCTIONS

def run_ttest(column, alternative="greater"):
    modern = df[df["type"] == "modern"][column]
    traditional = df[df["type"] == "traditional"][column]

    t_stat, p_value = ttest_ind(modern, traditional, alternative=alternative)

    print(f"\n===== Hypothesis Test for {column} =====")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")

    if p_value < 0.05:
        print("Result: Statistically significant difference.")
    else:
        print("Result: Not significant.")


# RUN MULTIPLE HYPOTHESIS TESTS


# 1 — Save Percentage
run_ttest("save_percentage", alternative="greater")

# 2 — Goals Prevented (PSxG-GA)
run_ttest("psxg_ga", alternative="greater")

# 3 — Successful Passes
run_ttest("successful_passes", alternative="greater")

# 4 — Errors Leading to Goal (modern expected fewer errors → alternative='less')
modern = df[df["type"] == "modern"]["errors_leading_to_goal"]
traditional = df[df["type"] == "traditional"]["errors_leading_to_goal"]

t_stat, p_value = ttest_ind(modern, traditional, alternative="less")

print("\n===== Hypothesis Test for Errors Leading to Goal =====")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
if p_value < 0.05:
    print("Result: Modern goalkeepers commit fewer errors (significant).")
else:
    print("Result: No significant difference.")

# FINISH

print("\nAnalysis completed successfully. All figures saved to /figures/")
