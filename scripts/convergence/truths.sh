# Slow- and fast-mixing truths of the composition-convergence study (Hydra values for
# +truths.<label>=...). Same numbers as SETS in scripts/convergence/sets.py.
# A row: death persist to_B div_AA div_AB div_BB; B row: death to_A persist div_AA div_AB div_BB.
declare -A TRUTH
TRUTH[S0]='{source:reference,seed:null,values:{A:{death:0.08,persist:0.10,to_B:0.08,div_AA:0.58,div_AB:0.10,div_BB:0.06},B:{death:0.20,to_A:0.01,persist:0.19,div_AA:0.005,div_AB:0.01,div_BB:0.585}}}'
TRUTH[S1]='{source:reference,seed:null,values:{A:{death:0.08,persist:0.10,to_B:0.10,div_AA:0.55,div_AB:0.12,div_BB:0.05},B:{death:0.20,to_A:0.02,persist:0.18,div_AA:0.01,div_AB:0.02,div_BB:0.57}}}'
TRUTH[S3]='{source:reference,seed:null,values:{A:{death:0.08,persist:0.10,to_B:0.10,div_AA:0.52,div_AB:0.12,div_BB:0.08},B:{death:0.15,to_A:0.05,persist:0.12,div_AA:0.02,div_AB:0.04,div_BB:0.62}}}'
TRUTH[F1]='{source:reference,seed:null,values:{A:{death:0.08,persist:0.10,to_B:0.10,div_AA:0.55,div_AB:0.12,div_BB:0.05},B:{death:0.20,to_A:0.30,persist:0.10,div_AA:0.10,div_AB:0.10,div_BB:0.20}}}'
