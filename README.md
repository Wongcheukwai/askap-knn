```markdown
# Collaborative Anomaly Detection in ASKAP Monitoring Data:
Integrating Machine Intelligence with Human Expertise

This repository contains Python scripts for processing various types of time-series data (PAF, DRX, BMF, Chiller) of ASKAP monitoring data and performing anomaly detection using machine learning techniques. The project is divided into three main components: data processing, machine learning methods for anomaly detection, and data visualization.

## Table of Contents

1. [Data Processing]
2. [Machine Learning Methods]
3. [Visualization]
4. [Getting Started]
5. [Usage]

## Data Processing

The data processing script (`data_processing.py`) handles different types of sensor data:

- PAF (Process Air Flow)
- DRX (Direct Expansion)
- BMF (Building Management Flow)
- Chiller

### Key Features:

- Uses `argparse` to select the processing mode (paf, drx, bmf, chiller, or combine)
- Reads CSV files from specified directories
- Processes data based on specific fields and conditions
- Combines data from multiple months or years
- Saves processed data in a structured format

### Modes:

1. **PAF Mode**:

   - Processes PAF data where only one card exists
   - Filters data based on unique field values
   - Combines data from multiple months in 2022

2. **DRX/BMF Mode**:

   - Processes DRX and BMF data
   - Filters data for specific fields ('bul_tempLocal_temp') and cards ('tempLocal2')
   - Combines data from multiple months in 2022

3. **Chiller Mode**:

   - Processes chiller data
   - Combines data from 2021 and 2022
   - Filters data for specific tables and chillers

4. **Combine Mode**:
   - Combines 2021 and 2022 CSV files
   - Groups files by prefix and combines them

## Machine Learning Methods

The machine learning script (`ml_pipeline_small_interval.py`) implements anomaly detection using the following techniques:

### Key Features:

- Processes data from multiple subsystems (chiller, BMF, DRX, PAF indoor, PAF outdoor)
- Uses K-Nearest Neighbors (KNN) for anomaly detection
- Implements custom thresholding based on mean distances
- Calculates severity scores for anomalies
- Identifies and ranks top anomalies
- Provides neighboring points information for context
- Generates visualizations of the time series data with marked anomalies

### Process:

1. Loads data for each subsystem
2. Filters data for a specific time range
3. Applies KNN algorithm to detect anomalies
4. Calculates anomaly severity and identifies top anomalies
5. Generates plots for each subsystem showing data points and anomalies
6. Saves anomaly information and plots

## Visualization

Two types of visualizations are provided:

### 1. Histogram (`visual_histogram.py`)

- Creates a heatmap of event counts across different subsystems and time intervals
- Uses seaborn for enhanced visual appeal
- Features:
  - Loads data from multiple subsystems
  - Filters data for a specific time range
  - Calculates event counts for each subsystem in defined time intervals
  - Generates a heatmap visualization

### 2. Line Chart (`visual_line.py`)

- Plots cumulative events for each subsystem over time
- Uses different markers for easy differentiation between subsystems
- Features:
  - Loads data from multiple subsystems
  - Filters data for a specific time range
  - Calculates cumulative event counts over time
  - Generates a line chart with distinct markers for each subsystem

## Getting Started

1. Clone this repository:
```

git clone https://github.com/Wongcheukwai/askap-knn.git
cd your-repo-name

```

2. Install the required dependencies (see [Dependencies](#dependencies))

3. Prepare your data files:
- Ensure your CSV files are in the correct format and located in the appropriate directories
- Update file paths in the scripts to match your directory structure

## Usage

### Data Processing

Run the data processing script with the desired mode:

```

python data_preprocessing.py --mode [paf|drx|bmf|chiller|combine]

```

Example:
```

python data_preprocessing.py --mode paf

```

### Anomaly Detection

Run the machine learning script:

```

python ml_pipeline_small_interval.py

```

Note: Ensure that the file paths and time ranges in the script match your data and requirements.

### Visualization

Run the visualization scripts:

```

python visual_histogram.py
python visual_line.py

```

Note: Update the file paths and time ranges in these scripts to match your processed data and analysis requirements.

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- adtk (Anomaly Detection Toolkit)


