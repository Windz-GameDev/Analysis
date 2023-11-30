import re
import pandas as pd
import os
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Define mappings for each cloud provider and benchmark
day_mappings = {
    'AWS': {
        'IOR': {'Nov15': '1', 'Nov16': '2', 'Nov21': '3', 'Nov22': '4'},
        'NPB': {'1115': '1', '1116': '2', '1121': '3', '1122': '4'}  # Adjust as needed for NPB
    },
    'GCP': {
        'IOR': {'Nov22': '1', 'Nov23': '2', 'Nov25': '3', 'Nov26': '4'},
        'NPB': {'Nov22': '1', 'Nov23': '2', 'Nov25': '3', 'Nov26': '4'}  # Adjust as needed for NPB
    }
}

def parse_file_name(file_name, cloud_provider, benchmark):
    # Choose the appropriate map based on cloud provider and benchmark
    date_to_day = day_mappings.get(cloud_provider, {}).get(benchmark, {})

    if benchmark == 'NPB' and cloud_provider == 'AWS':
        # AWS NPB file naming pattern
        npb_match = re.search(r'(\d{4})_(\d+)_n(\d+)', file_name)
        if npb_match:
            date = npb_match.group(1)
            day = date_to_day.get(date, 'Unknown')
            test_iteration = npb_match.group(2)
            num_nodes = npb_match.group(3)
    else:
        # IOR file naming pattern (for AWS IOR and all GCP files, including GCP NPB)
        ior_match = re.search(r'(\w+)-(\d+)\.(\d+)node', file_name)
        if ior_match:
            month_day = ior_match.group(1)
            day = date_to_day.get(month_day, 'Unknown')
            test_iteration = ior_match.group(2)
            num_nodes = ior_match.group(3)

    return {
        'Day': day,
        'Test Iteration': test_iteration,
        'Number of Nodes': num_nodes
    }

# Function to extract IOR metrics from a file
def extract_ior_metrics(file_path, keywords):
    data = {}
    with open(file_path, 'r') as file:
        content = file.read()  # Read the entire content of the file
        for keyword in keywords:
            match = re.search(rf"{keyword}\s*:\s*([\d.]+)", content)
            if match:
                data[keyword] = float(match.group(1))
    return data

# Function to extract NPB metrics from a file
def extract_npb_metrics(file_path, keywords):
    data = {}
    with open(file_path, 'r') as file:
        content = file.readlines()  # Read the entire content of the file as lines
        for line in content:
            for keyword in keywords:
                match = re.search(rf"{keyword}\s*=\s*([\d.]+)", line)
                if match:
                    data[keyword] = float(match.group(1))
    return data

# Function to process a directory of files and extract the metrics
def process_directory(directory, cloud_provider, ior_keywords, npb_keywords):
    ior_data = []
    npb_data = []
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            benchmark = 'IOR' if 'IOR' in root else 'NPB'
            file_info = parse_file_name(file_name, cloud_provider, benchmark)
            if 'IOR' in root:
                data = extract_ior_metrics(file_path, ior_keywords)
            elif 'NPB' in root:
                data = extract_npb_metrics(file_path, npb_keywords)
            data.update(file_info)
            data['Cloud Provider'] = cloud_provider
            if 'IOR' in root:
                ior_data.append(data)
            elif 'NPB' in root:
                npb_data.append(data)
    return ior_data, npb_data

# Function to add descriptive statistics to a boxplot
def add_stats_to_boxplot(box_plot, df, y):
    for i in range(len(box_plot.get_xticklabels())):
        for j, provider in enumerate(['AWS', 'GCP']):
            stats = df[(df['Day'] == box_plot.get_xticklabels()[i].get_text()) & (df['Cloud Provider'] == provider)][y].describe()
            iqr = stats['75%'] - stats['25%']
            box_plot.text(i - 0.2 + j * 0.4, box_plot.get_ylim()[0], f"{provider}\nMin: {stats['min']:.2f}\nQ1: {stats['25%']:.2f}\nMedian: {stats['50%']:.2f}\nQ3: {stats['75%']:.2f}\nMax: {stats['max']:.2f}\nIQR: {iqr:.2f}", verticalalignment='top', horizontalalignment='center', fontsize=8)

# Improved function to space AWS and GCP provider statistics further apart
def add_stats_to_boxplot(box_plot, df, y, ax, fig):
    num_days = len(df['Day'].unique())
    for i, day in enumerate(sorted(df['Day'].unique())):
        day_data = df[df['Day'] == day]
        for j, provider in enumerate(['AWS', 'GCP']):
            provider_data = day_data[day_data['Cloud Provider'] == provider]
            stats = provider_data[y].describe()
            if stats.count() > 0:  # Check if there are any values to describe
                iqr = stats['75%'] - stats['25%']
                stats_text = f"{provider}\nMin: {stats['min']:.2f}\n1Q: {stats['25%']:.2f}\n" \
                             f"Median: {stats['50%']:.2f}\n3Q: {stats['75%']:.2f}\n" \
                             f"Max: {stats['max']:.2f}\nIQR: {iqr:.2f}"
                # Adjust the horizontal position for each provider
                x_position = (i / num_days) + (j * 0.12)  # Spacing between providers
                fig.text(x_position, -0.15, stats_text,  # Adjust vertical position
                         verticalalignment='top', horizontalalignment='left', fontsize=8, family='monospace',
                         transform=ax.transAxes)

# Function to annotate bar plots with t-test results
def annotate_ttest_results(ax, t_test_result, offset_below_title=0.2):
    # Annotate with existing t-test results
    ax.text(0.5, 1 - offset_below_title, f"T-test Results:\nT-statistic = {t_test_result['T-statistic']:.2f}\nP-value = {t_test_result['P-value']:.4f}",
            horizontalalignment='center', fontsize=10, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))

# Helper function to sanitize metric names for file names
def sanitize_filename(name):
    return name.replace(" ", "_").replace("/", "_")

# Function to calculate Coefficient of Variation (COV)
def calculate_cov(df, mean_col, std_col):
    return (df[std_col] / df[mean_col]) * 100

# Define the keywords for IOR and NPB metrics
ior_keywords = ['Max Write', 'Max Read']
npb_keywords = ['Mop/s total', 'Time in seconds']

# Process the directories for AWS and GCP
ior_data_aws, npb_data_aws = process_directory('AWS', 'AWS', ior_keywords, npb_keywords)
ior_data_gcp, npb_data_gcp = process_directory('GCP', 'GCP', ior_keywords, npb_keywords)

# Combine the data from AWS and GCP into single DataFrames
combined_ior_df = pd.DataFrame(ior_data_aws + ior_data_gcp)
combined_npb_df = pd.DataFrame(npb_data_aws + npb_data_gcp)

# Convert 'Day', 'Test Iteration', 'Number of Nodes', and metric columns to numeric types
combined_ior_df['Day'] = pd.to_numeric(combined_ior_df['Day'], errors='coerce')
combined_ior_df['Test Iteration'] = pd.to_numeric(combined_ior_df['Test Iteration'], errors='coerce')
combined_ior_df['Number of Nodes'] = pd.to_numeric(combined_ior_df['Number of Nodes'], errors='coerce')
combined_ior_df['Max Write'] = pd.to_numeric(combined_ior_df['Max Write'], errors='coerce')
combined_ior_df['Max Read'] = pd.to_numeric(combined_ior_df['Max Read'], errors='coerce')

combined_npb_df['Day'] = pd.to_numeric(combined_npb_df['Day'], errors='coerce')
combined_npb_df['Test Iteration'] = pd.to_numeric(combined_npb_df['Test Iteration'], errors='coerce')
combined_npb_df['Number of Nodes'] = pd.to_numeric(combined_npb_df['Number of Nodes'], errors='coerce')
combined_npb_df['Mop/s total'] = pd.to_numeric(combined_npb_df['Mop/s total'], errors='coerce')
combined_npb_df['Time in seconds'] = pd.to_numeric(combined_npb_df['Time in seconds'], errors='coerce')

# Drop rows with NaN values that were created during conversion
combined_ior_df.dropna(inplace=True)
combined_npb_df.dropna(inplace=True)

# Reorder columns to make 'Cloud Provider' and 'Number of Nodes' first
ior_columns = ['Cloud Provider', 'Number of Nodes'] + [col for col in combined_ior_df.columns if col not in ['Cloud Provider', 'Number of Nodes']]
npb_columns = ['Cloud Provider', 'Number of Nodes'] + [col for col in combined_npb_df.columns if col not in ['Cloud Provider', 'Number of Nodes']]

combined_ior_df = combined_ior_df[ior_columns]
combined_npb_df = combined_npb_df[npb_columns]

# If we need to print the whole dataset
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

# Display the first few rows of the DataFrames
# print(combined_ior_df)
# print(combined_npb_df)

avg_std_ior_metrics_df = (combined_ior_df
                          .groupby(['Cloud Provider', 'Number of Nodes'])
                          .agg({'Max Write': ['mean', 'std'], 'Max Read': ['mean', 'std']})
                          .reset_index())
avg_std_ior_metrics_df.columns = ['Cloud Provider', 'Number of Nodes', 'Average Max Write', 'Std Dev Max Write', 'Average Max Read', 'Std Dev Max Read']

# Adding COV to IOR metrics dataframe
avg_std_ior_metrics_df['COV Max Write (%)'] = calculate_cov(avg_std_ior_metrics_df, 'Average Max Write', 'Std Dev Max Write')
avg_std_ior_metrics_df['COV Max Read (%)'] = calculate_cov(avg_std_ior_metrics_df, 'Average Max Read', 'Std Dev Max Read')

# Reorder columns for avg_std_ior_metrics_df
avg_std_ior_metrics_df = avg_std_ior_metrics_df[['Cloud Provider', 'Number of Nodes', 
                                                 'Average Max Write', 'Std Dev Max Write', 'COV Max Write (%)', 
                                                 'Average Max Read', 'Std Dev Max Read', 'COV Max Read (%)']]

print(f"\nIOR Averages, Standard Deviations and COV\n{avg_std_ior_metrics_df}")

# Splitting the data into AWS and GCP groups
aws_avg_write = avg_std_ior_metrics_df[(avg_std_ior_metrics_df['Cloud Provider'] == 'AWS')]['Average Max Write']
gcp_avg_write = avg_std_ior_metrics_df[(avg_std_ior_metrics_df['Cloud Provider'] == 'GCP')]['Average Max Write']
aws_avg_read = avg_std_ior_metrics_df[(avg_std_ior_metrics_df['Cloud Provider'] == 'AWS')]['Average Max Read']
gcp_avg_read = avg_std_ior_metrics_df[(avg_std_ior_metrics_df['Cloud Provider'] == 'GCP')]['Average Max Read']

# Performing t-tests
avg_write_t_test = stats.ttest_rel(aws_avg_write, gcp_avg_write)  # T-test for Average Max Write
avg_read_t_test = stats.ttest_rel(aws_avg_read, gcp_avg_read)    # T-test for Average Max Read

# Outputting the results
avg_write_t_test_result = {"T-statistic": avg_write_t_test.statistic, "P-value": avg_write_t_test.pvalue}
avg_read_t_test_result = {"T-statistic": avg_read_t_test.statistic, "P-value": avg_read_t_test.pvalue}

# Print the results
print("\nIOR Benchmark - T-test Results for Average Max Write:")
print(avg_write_t_test_result)
print("\nIOR Benchmark - T-test Results for Average Max Read:")
print(avg_read_t_test_result)

# Process NPB data
combined_npb_df['Mop/s total'] = pd.to_numeric(combined_npb_df['Mop/s total'], errors='coerce')
combined_npb_df['Time in seconds'] = pd.to_numeric(combined_npb_df['Time in seconds'], errors='coerce')
combined_npb_df['Number of Nodes'] = pd.to_numeric(combined_npb_df['Number of Nodes'], errors='coerce')

# Drop rows with NaN values if any were created during conversion
combined_npb_df = combined_npb_df.dropna()

avg_std_npb_metrics_df = (combined_npb_df
                          .groupby(['Cloud Provider', 'Number of Nodes'])
                          .agg({'Mop/s total': ['mean', 'std'], 'Time in seconds': ['mean', 'std']})
                          .reset_index())
avg_std_npb_metrics_df.columns = ['Cloud Provider', 'Number of Nodes', 'Average Mop/s', 'Std Dev Mop/s', 'Average Time', 'Std Dev Time']

# Adding COV to NPB metrics dataframe
avg_std_npb_metrics_df['COV Mop/s (%)'] = calculate_cov(avg_std_npb_metrics_df, 'Average Mop/s', 'Std Dev Mop/s')
avg_std_npb_metrics_df['COV Time (%)'] = calculate_cov(avg_std_npb_metrics_df, 'Average Time', 'Std Dev Time')

# Reorder columns for avg_std_npb_metrics_df
avg_std_npb_metrics_df = avg_std_npb_metrics_df[['Cloud Provider', 'Number of Nodes', 
                                                 'Average Mop/s', 'Std Dev Mop/s', 'COV Mop/s (%)', 
                                                 'Average Time', 'Std Dev Time', 'COV Time (%)']]

print(f"\nNPB Averages, Standard Deviations and COV\n{avg_std_npb_metrics_df}")

# Splitting the data into AWS and GCP groups for NPB
aws_avg_mops = avg_std_npb_metrics_df[(avg_std_npb_metrics_df['Cloud Provider'] == 'AWS')]['Average Mop/s']
gcp_avg_mops = avg_std_npb_metrics_df[(avg_std_npb_metrics_df['Cloud Provider'] == 'GCP')]['Average Mop/s']
aws_avg_time = avg_std_npb_metrics_df[(avg_std_npb_metrics_df['Cloud Provider'] == 'AWS')]['Average Time']
gcp_avg_time = avg_std_npb_metrics_df[(avg_std_npb_metrics_df['Cloud Provider'] == 'GCP')]['Average Time']

# Performing t-tests for NPB benchmark
npb_mops_t_test = stats.ttest_rel(aws_avg_mops, gcp_avg_mops)  # T-test for Average Mop/s
npb_time_t_test = stats.ttest_rel(aws_avg_time, gcp_avg_time)  # T-test for Average Time

# Outputting the results
npb_mops_t_test_result = {"T-statistic": npb_mops_t_test.statistic, "P-value": npb_mops_t_test.pvalue}
npb_time_t_test_result = {"T-statistic": npb_time_t_test.statistic, "P-value": npb_time_t_test.pvalue}

# Print the results
print("\nNPB Benchmark - T-test Results for Average Mop/s:")
print(npb_mops_t_test_result)
print("\nNPB Benchmark - T-test Results for Average Time:")
print(f"{npb_time_t_test_result}\n")

# Create a results directory if it doesn't exist
results_directory = 'results'
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

# Setting the style for the plots
sns.set(style="whitegrid")

# Create bar plots and add t-test results
plt.figure(figsize=(14, 6))

# Plot for IOR Average Max Write
ax1 = plt.subplot(1, 2, 1)
sns.barplot(x='Number of Nodes', y='Average Max Write', hue='Cloud Provider', data=avg_std_ior_metrics_df)
ax1.set_title('IOR Benchmark - Average Max Write', fontsize=12)
ax1.set_xlabel('Number of Nodes')
ax1.set_ylabel('Average Max Write (MiB/sec)')
annotate_ttest_results(ax1, avg_write_t_test_result)

# Plot for IOR Average Max Read
ax2 = plt.subplot(1, 2, 2)
sns.barplot(x='Number of Nodes', y='Average Max Read', hue='Cloud Provider', data=avg_std_ior_metrics_df)
ax2.set_title('IOR Benchmark - Average Max Read', fontsize=12)
ax2.set_xlabel('Number of Nodes')
ax2.set_ylabel('Average Max Read (MiB/sec)')
annotate_ttest_results(ax2, avg_read_t_test_result)

plt.tight_layout()
plt.savefig(os.path.join(results_directory, 'IOR_Benchmark_Plots.png'))
plt.close()

plt.figure(figsize=(14, 6))

# Plot for NPB Average Mop/s
ax3 = plt.subplot(1, 2, 1)
sns.barplot(x='Number of Nodes', y='Average Mop/s', hue='Cloud Provider', data=avg_std_npb_metrics_df)
ax3.set_title('NPB Benchmark - Average Mop/s', fontsize=12)
ax3.set_xlabel('Number of Nodes')
ax3.set_ylabel('Average Mop/s')
annotate_ttest_results(ax3, npb_mops_t_test_result)

# Plot for NPB Average Time
ax4 = plt.subplot(1, 2, 2)
sns.barplot(x='Number of Nodes', y='Average Time', hue='Cloud Provider', data=avg_std_npb_metrics_df)
ax4.set_title('NPB Benchmark - Average Time', fontsize=12)
ax4.set_xlabel('Number of Nodes')
ax4.set_ylabel('Average Time (Seconds)')
annotate_ttest_results(ax4, npb_time_t_test_result)

plt.tight_layout()
plt.savefig(os.path.join(results_directory, 'NPB_Benchmark_Plots.png'))
plt.close()

boxplots_directory = os.path.join(results_directory, 'boxplots')
if not os.path.exists(boxplots_directory):
    os.makedirs(boxplots_directory)

# Define a larger figure size for the box plots
figsize = (12, 12)  # Adjust this size as needed

# Boxplot for IOR Read
plt.figure(figsize=figsize)
ax = plt.gca()
fig = plt.gcf()
box_plot = sns.boxplot(x='Day', y='Max Write', hue='Cloud Provider', data=combined_ior_df, ax=ax)
plt.title('IOR Benchmark - Write Performance by Day')
plt.xlabel('Day')
plt.ylabel('Max Write')
add_stats_to_boxplot(box_plot, combined_ior_df, 'Max Write', ax, fig)
plt.subplots_adjust(bottom=0.3)  # Increase bottom margin to make room for text
plt.savefig(os.path.join(results_directory, 'boxplots', 'IOR_Benchmark_Write.png'))
plt.close()

# Boxplot for IOR Read
plt.figure(figsize=figsize)
ax = plt.gca()
fig = plt.gcf()
box_plot = sns.boxplot(x='Day', y='Max Read', hue='Cloud Provider', data=combined_ior_df, ax=ax)
plt.title('IOR Benchmark - Read Performance by Day')
plt.xlabel('Day')
plt.ylabel('Max Read')
add_stats_to_boxplot(box_plot, combined_ior_df, 'Max Read', ax, fig)
plt.subplots_adjust(bottom=0.3)  # Increase bottom margin to make room for text
plt.savefig(os.path.join(results_directory, 'boxplots', 'IOR_Benchmark_Read.png'))
plt.close()

# Boxplot for NPB Mop/s
plt.figure(figsize=figsize)
ax = plt.gca()
fig = plt.gcf()
box_plot = sns.boxplot(x='Day', y='Mop/s total', hue='Cloud Provider', data=combined_npb_df, ax=ax)
plt.title('NPB Benchmark - Mop/s Performance by Day')
plt.xlabel('Day')
plt.ylabel('Mop/s total')
add_stats_to_boxplot(box_plot, combined_npb_df, 'Mop/s total', ax, fig)
plt.subplots_adjust(bottom=0.3)
plt.savefig(os.path.join(results_directory, 'boxplots', 'NPB_Benchmark_Mops.png'))
plt.close()

# Boxplot for NPB Time
plt.figure(figsize=figsize)
ax = plt.gca()
fig = plt.gcf()
box_plot = sns.boxplot(x='Day', y='Time in seconds', hue='Cloud Provider', data=combined_npb_df, ax=ax)
plt.title('NPB Benchmark - Time Performance by Day')
plt.xlabel('Day')
plt.ylabel('Time in seconds')
add_stats_to_boxplot(box_plot, combined_npb_df, 'Time in seconds', ax, fig)
plt.subplots_adjust(bottom=0.3)
plt.savefig(os.path.join(results_directory, 'boxplots', 'NPB_Benchmark_Time.png'))
plt.close()

# Loop through metrics for IOR benchmark
for metric_to_plot in ['Max Write', 'Max Read']:

    # Setting the style for the plots
    sns.set(style="whitegrid")

    barplot_data = combined_ior_df[['Cloud Provider', 'Number of Nodes', 'Day', metric_to_plot]].copy()
    barplot_data['Group'] = barplot_data['Cloud Provider'] + barplot_data['Number of Nodes'].astype(str) + 'Node'

    plt.figure(figsize=(15, 6))
    sns.barplot(x='Group', y=metric_to_plot, hue='Day', data=barplot_data, palette='deep', errorbar=None)
    plt.xlabel('Group')
    plt.ylabel(metric_to_plot)
    plt.title(f'IOR Benchmark - {metric_to_plot} by Group and Day')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Dynamic file name
    file_name = f'IOR_{metric_to_plot.replace(" ", "_")}_Grouped_Barplot.png'
    plt.savefig(os.path.join(results_directory, file_name))
    plt.close()

# Loop through metrics for NPB benchmark
for metric_to_plot in ['Mop/s total', 'Time in seconds']:

    # Setting the style for the plots
    sns.set(style="whitegrid")

    barplot_data = combined_npb_df[['Cloud Provider', 'Number of Nodes', 'Day', metric_to_plot]].copy()
    barplot_data['Group'] = barplot_data['Cloud Provider'] + barplot_data['Number of Nodes'].astype(str) + 'Node'

    plt.figure(figsize=(15, 6))
    sns.barplot(x='Group', y=metric_to_plot, hue='Day', data=barplot_data, palette='deep', errorbar=None)
    plt.xlabel('Group')
    plt.ylabel(metric_to_plot)
    plt.title(f'NPB Benchmark - {metric_to_plot} by Group and Day')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Use the sanitize_filename function for the file name
    file_name = f'NPB_{sanitize_filename(metric_to_plot)}_Grouped_Barplot.png'
    plt.savefig(os.path.join(results_directory, file_name))
    plt.close()

# Create a datasets directory if it doesn't exist
datasets_directory = 'datasets'
if not os.path.exists(datasets_directory):
    os.makedirs(datasets_directory)

# Rename columns in combined_ior_df for clarity
combined_ior_df.columns = [
    'Cloud Provider', 
    'Number of Nodes', 
    'Max Write (MiB/sec)', 
    'Max Read (MiB/sec)', 
    'Day', 
    'Test Iteration'
]

# Correctly renaming columns in avg_std_npb_metrics_df after adding COV columns
avg_std_npb_metrics_df.columns = [
    'Cloud Provider', 
    'Number of Nodes', 
    'Average Mop/s', 
    'Std Dev Mop/s', 
    'COV Mop/s (%)', 
    'Average Time', 
    'Std Dev Time',
    'COV Time (%)'
]

# Save the DataFrames to CSV in the datasets directory
avg_std_ior_metrics_df.to_csv(os.path.join(datasets_directory, 'avg_std_ior_metrics.csv'), index=False)
avg_std_npb_metrics_df.to_csv(os.path.join(datasets_directory, 'avg_std_npb_metrics.csv'), index=False)
combined_ior_df.to_csv(os.path.join(datasets_directory, 'combined_ior_df.csv'), index=False)
combined_npb_df.to_csv(os.path.join(datasets_directory, 'combined_npb_df.csv'), index=False)   