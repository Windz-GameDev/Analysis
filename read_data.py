import re
import pandas as pd
import os

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

# Display the first few rows of the DataFrames
print(combined_ior_df)
print(combined_npb_df)
