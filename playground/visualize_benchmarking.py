# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Function to load data from JSONL file
def load_data(filename):
    return pd.read_json(filename, lines=True)


# Load the datasets
df_seq_nolimit = load_data("../results_sequential_maxiter_no_limit.jsonl")
df_seq_200 = load_data("../results_sequential_maxiter_200.jsonl")
df_st_nolimit = load_data("../results_stacking_maxiter_no_limit.jsonl")
df_st_200 = load_data("../results_stacking_maxiter_200.jsonl")


# Merge the dataframes on batch_size and dimension
# We will use df_seq_nolimit as the base for comparison
merged_df_seq_200 = pd.merge(
    df_seq_nolimit,
    df_seq_200,
    on=["batch_size", "dimension"],
    suffixes=("_seq_nolimit", "_seq_200"),
)
merged_df_st_nolimit = pd.merge(
    df_seq_nolimit,
    df_st_nolimit,
    on=["batch_size", "dimension"],
    suffixes=("_seq_nolimit", "_st_nolimit"),
)
merged_df_st_200 = pd.merge(
    df_seq_nolimit,
    df_st_200,
    on=["batch_size", "dimension"],
    suffixes=("_seq_nolimit", "_st_200"),
)

# Calculate the ratio of best_f_opts
merged_df_seq_200["ratio"] = (
    merged_df_seq_200["best_f_opts_seq_200"]
    / merged_df_seq_200["best_f_opts_seq_nolimit"]
)
merged_df_st_nolimit["ratio"] = (
    merged_df_st_nolimit["best_f_opts_st_nolimit"]
    / merged_df_st_nolimit["best_f_opts_seq_nolimit"]
)
merged_df_st_200["ratio"] = (
    merged_df_st_200["best_f_opts_st_200"] / merged_df_st_200["best_f_opts_seq_nolimit"]
)

# Plotting the distributions of the ratios
plt.figure(figsize=(10, 6))
sns.kdeplot(
    merged_df_seq_200["ratio"],
    label="Sequential maxiter=200 / Sequential maxiter=no_limit",
    fill=True,
)
sns.kdeplot(
    merged_df_st_nolimit["ratio"],
    label="Stacking maxiter=no_limit / Sequential maxiter=no_limit",
    fill=True,
)
sns.kdeplot(
    merged_df_st_200["ratio"],
    label="Stacking maxiter=200 / Sequential maxiter=no_limit",
    fill=True,
)

plt.title("Distribution of best_f_opts Ratio (File j / File 1)")
plt.xlabel("Ratio of best_f_opts")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.xlim(0, 2)
plt.savefig("best_f_opt_ratio_distribution.png")
plt.close()

print("Plot saved to best_f_opt_ratio_distribution.png")


# %%
# 比率データをまとめたDataFrameを作成
ratio_df = pd.DataFrame(
    {
        "Sequential maxiter=200": merged_df_seq_200["ratio"],
        "Stacking maxiter=no_limit": merged_df_st_nolimit["ratio"],
        "Stacking maxiter=200": merged_df_st_200["ratio"],
    }
)
# 横長のDataFrameから縦長に変換
ratio_df_melted = ratio_df.melt(var_name="Method", value_name="Ratio")

plt.figure(figsize=(10, 6))
sns.boxplot(x="Ratio", y="Method", data=ratio_df_melted)
plt.title("Comparison of best_f_opts Ratios")
plt.xscale("log")
plt.grid(True)
plt.xlim(0.8, 1.5)
plt.savefig("ratio_boxplot.png")
