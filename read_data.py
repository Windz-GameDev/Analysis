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

# Define the keywords for IOR and NPB metrics
ior_keywords = ['Max Write', 'Max Read']
npb_keywords = ['Mop/s total', 'Time in seconds']

# Process the directories for AWS and GCP
ior_data_aws, npb_data_aws = process_directory('AWS', 'AWS', ior_keywords, npb_keywords)
ior_data_gcp, npb_data_gcp = process_directory('GCP', 'GCP', ior_keywords, npb_keywords)

# Combine the data from AWS and GCP into single DataFrames
combined_ior_df = pd.DataFrame(ior_data_aws + ior_data_gcp)
combined_npb_df = pd.DataFrame(npb_data_aws + npb_data_gcp)

# Reorder columns to make 'Cloud Provider' and 'Number of Nodes' first
ior_columns = ['Cloud Provider', 'Number of Nodes'] + [col for col in combined_ior_df.columns if col not in ['Cloud Provider', 'Number of Nodes']]
npb_columns = ['Cloud Provider', 'Number of Nodes'] + [col for col in combined_npb_df.columns if col not in ['Cloud Provider', 'Number of Nodes']]

combined_ior_df = combined_ior_df[ior_columns]
combined_npb_df = combined_npb_df[npb_columns]


# If we need to print the whole dataset
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

# Display the first few rows of the DataFrames
print(combined_ior_df)
print(combined_npb_df)

# Convert relevant columns to numeric types
combined_ior_df['Max Write'] = pd.to_numeric(combined_ior_df['Max Write'], errors='coerce')
combined_ior_df['Max Read'] = pd.to_numeric(combined_ior_df['Max Read'], errors='coerce')
combined_ior_df['Number of Nodes'] = pd.to_numeric(combined_ior_df['Number of Nodes'], errors='coerce')

# Drop rows with NaN values if any were created during conversion
combined_ior_df = combined_ior_df.dropna()

# Calculate averages for each node configuration for both AWS and GCP
avg_ior_metrics_df = combined_ior_df.groupby(['Cloud Provider', 'Number of Nodes']).agg({'Max Write': 'mean', 'Max Read': 'mean'}).reset_index()

avg_ior_metrics_df.rename(columns={'Max Write': 'Average Max Write', 'Max Read': 'Average Max Read'}, inplace=True)
print("\n", avg_ior_metrics_df)

# Splitting the data into AWS and GCP groups
aws_avg_write = avg_ior_metrics_df[(avg_ior_metrics_df['Cloud Provider'] == 'AWS')]['Average Max Write']
gcp_avg_write = avg_ior_metrics_df[(avg_ior_metrics_df['Cloud Provider'] == 'GCP')]['Average Max Write']
aws_avg_read = avg_ior_metrics_df[(avg_ior_metrics_df['Cloud Provider'] == 'AWS')]['Average Max Read']
gcp_avg_read = avg_ior_metrics_df[(avg_ior_metrics_df['Cloud Provider'] == 'GCP')]['Average Max Read']

# Performing t-tests
avg_write_t_test = stats.ttest_ind(aws_avg_write, gcp_avg_write, equal_var=False)  # T-test for Average Max Write
avg_read_t_test = stats.ttest_ind(aws_avg_read, gcp_avg_read, equal_var=False)    # T-test for Average Max Read

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

# Calculate averages for each node configuration for both AWS and GCP
avg_npb_metrics_df = combined_npb_df.groupby(['Cloud Provider', 'Number of Nodes']).agg({'Mop/s total': 'mean', 'Time in seconds': 'mean'}).reset_index()

avg_npb_metrics_df.rename(columns={'Mop/s total': 'Average Mop/s', 'Time in seconds': 'Average Time'}, inplace=True)
print("\n", avg_npb_metrics_df)

# Splitting the data into AWS and GCP groups for NPB
aws_avg_mops = avg_npb_metrics_df[(avg_npb_metrics_df['Cloud Provider'] == 'AWS')]['Average Mop/s']
gcp_avg_mops = avg_npb_metrics_df[(avg_npb_metrics_df['Cloud Provider'] == 'GCP')]['Average Mop/s']
aws_avg_time = avg_npb_metrics_df[(avg_npb_metrics_df['Cloud Provider'] == 'AWS')]['Average Time']
gcp_avg_time = avg_npb_metrics_df[(avg_npb_metrics_df['Cloud Provider'] == 'GCP')]['Average Time']

# Performing t-tests for NPB benchmark
npb_mops_t_test = stats.ttest_ind(aws_avg_mops, gcp_avg_mops, equal_var=False)  # T-test for Average Mop/s
npb_time_t_test = stats.ttest_ind(aws_avg_time, gcp_avg_time, equal_var=False)  # T-test for Average Time

# Outputting the results
npb_mops_t_test_result = {"T-statistic": npb_mops_t_test.statistic, "P-value": npb_mops_t_test.pvalue}
npb_time_t_test_result = {"T-statistic": npb_time_t_test.statistic, "P-value": npb_time_t_test.pvalue}

# Print the results
print("\nNPB Benchmark - T-test Results for Average Mop/s:")
print(npb_mops_t_test_result)
print("\nNPB Benchmark - T-test Results for Average Time:")
print(npb_time_t_test_result)
print("\n")

# Create a results directory if it doesn't exist
results_directory = 'results'
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

# Setting the style for the plots
sns.set(style="whitegrid")

# Save the plots
plt.figure(figsize=(14, 6))
# Plot for Average Max Write
plt.subplot(1, 2, 1)
sns.barplot(x='Number of Nodes', y='Average Max Write', hue='Cloud Provider', data=avg_ior_metrics_df)
plt.title('IOR Benchmark - Average Max Write')
plt.xlabel('Number of Nodes')
plt.ylabel('Average Max Write')
# Plot for Average Max Read
plt.subplot(1, 2, 2)
sns.barplot(x='Number of Nodes', y='Average Max Read', hue='Cloud Provider', data=avg_ior_metrics_df)
plt.title('IOR Benchmark - Average Max Read')
plt.xlabel('Number of Nodes')
plt.ylabel('Average Max Read')
plt.tight_layout()
plt.savefig(os.path.join(results_directory, 'IOR_Benchmark_Plots.png'))

plt.figure(figsize=(14, 6))
# Plot for Average Mop/s
plt.subplot(1, 2, 1)
sns.barplot(x='Number of Nodes', y='Average Mop/s', hue='Cloud Provider', data=avg_npb_metrics_df)
plt.title('NPB Benchmark - Average Mop/s')
plt.xlabel('Number of Nodes')
plt.ylabel('Average Mop/s')
# Plot for Average Time
plt.subplot(1, 2, 2)
sns.barplot(x='Number of Nodes', y='Average Time', hue='Cloud Provider', data=avg_npb_metrics_df)
plt.title('NPB Benchmark - Average Time')
plt.xlabel('Number of Nodes')
plt.ylabel('Average Time (Seconds)')
plt.tight_layout()
plt.savefig(os.path.join(results_directory, 'NPB_Benchmark_Plots.png'))

# Create a datasets directory if it doesn't exist
datasets_directory = 'datasets'
if not os.path.exists(datasets_directory):
    os.makedirs(datasets_directory)

# Save the DataFrames to CSV in the datasets directory
avg_ior_metrics_df.to_csv(os.path.join(datasets_directory, 'avg_ior_metrics.csv'), index=False)
avg_npb_metrics_df.to_csv(os.path.join(datasets_directory, 'avg_npb_metrics.csv'), index=False)
combined_ior_df.to_csv(os.path.join(datasets_directory, 'combined_ior_df.csv'), index=False)
combined_npb_df.to_csv(os.path.join(datasets_directory, 'combined_npb_df.csv'), index=False)   

'''

# Calculate the mean for each day across all iterations for IOR
ior_day_mean = combined_ior_df.groupby(['Cloud Provider', 'Day']).mean().reset_index()

# Create box plots for IOR with Day on x-axis and metrics on y-axis
plt.figure(figsize=(14, 6))

# Boxplot for IOR Write
plt.subplot(1, 2, 1)
sns.boxplot(x='Day', y='Max Write', hue='Cloud Provider', data=ior_day_mean)
plt.title('IOR Benchmark - Write Performance by Day')
plt.xlabel('Day')
plt.ylabel('Max Write')

# Boxplot for IOR Read
plt.subplot(1, 2, 2)
sns.boxplot(x='Day', y='Max Read', hue='Cloud Provider', data=ior_day_mean)
plt.title('IOR Benchmark - Read Performance by Day')
plt.xlabel('Day')
plt.ylabel('Max Read')

plt.tight_layout()
plt.show()

# Calculate the mean for each day across all iterations for NPB
npb_day_mean = combined_npb_df.groupby(['Cloud Provider', 'Day']).mean().reset_index()

# Create box plots for NPB with Day on x-axis and metrics on y-axis
plt.figure(figsize=(14, 6))

# Boxplot for NPB Mop/s
plt.subplot(1, 2, 1)
sns.boxplot(x='Day', y='Mop/s total', hue='Cloud Provider', data=npb_day_mean)
plt.title('NPB Benchmark - Mop/s Performance by Day')
plt.xlabel('Day')
plt.ylabel('Mop/s total')

# Boxplot for NPB Time
plt.subplot(1, 2, 2)
sns.boxplot(x='Day', y='Time in seconds', hue='Cloud Provider', data=npb_day_mean)
plt.title('NPB Benchmark - Time Performance by Day')
plt.xlabel('Day')
plt.ylabel('Time in seconds')

plt.tight_layout()
plt.show()

'''