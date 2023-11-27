# Cloud-Computing---Benchmarks-Analysis

This repository is part of Project 2 for CEN6086 at the University of North Florida, focusing on analyzing benchmark results from AWS and GCP for the IOR and NPB benchmarks.

## Introduction

The primary goal of this project is to conduct a comprehensive analysis of performance benchmarks conducted on AWS and GCP cloud platforms. We utilize the IOR and NPB benchmarks to evaluate the performance metrics and draw comparative insights.

## Repository Structure

- `AWS/`: Contains data files and scripts related to AWS benchmarks.
- `GCP/`: Contains data files and scripts related to GCP benchmarks.
- `datasets/`: Directory where processed datasets are stored.
- `results/`: Contains generated graphs and result files from the analysis.
- `read_data.py`: The main Python script for data processing and analysis.
- `README.md`: This file.

## Getting Started

To use this repository for analyzing benchmark data, follow these steps:

### Prerequisites

Ensure you have Python installed on your system. The analysis script relies on several Python libraries such as Pandas, Matplotlib, and Seaborn. Install these packages using pip:

```bash
pip install pandas matplotlib seaborn scipy
```
### Running the Script
Clone the repository to your local machine:

```bash
git clone https://github.com/Windz-GameDev/Cloud-Computing---Benchmarks-Analysis.git
```
Navigate to the cloned repository directory.

Place your benchmark data files in the respective AWS/ and GCP/ directories. Ensure your naming schemes are the same as the example files.

Run the read_data.py script:

bash
Copy code
python read_data.py
The script will process the data and save the analysis results in the datasets/ and results/ directories.

## Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License
This project is open-source and available under the MIT License.

## Contact
For any queries regarding this project, please reach out to n01421643@unf.edu.
