"""Optional report-oriented analyses.

Core reproducibility is handled by main.py. Run this after main.py when
report figures/diagnostics are also needed.
"""

from analysis.abnormal3_analysis import run_abnormal3_analysis
from analysis.dda import run_dda
from analysis.eda import run_eda
from analysis.error_analysis import run_error_analysis
from analysis.gap_sensitivity import run_gap_sensitivity
from analysis.influence_analysis import run_influence_analysis


def main():
    run_dda()
    run_gap_sensitivity()
    run_eda()
    run_influence_analysis()
    run_error_analysis()
    run_abnormal3_analysis()
    print("\nOPTIONAL ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
