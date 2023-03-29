#code for mediation analysis

import pandas as pd
from mediation.utils import ModelSpecificationError
from mediation import Mediation

data_dir = '/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/'
space = 'MNI'

# Specify the names of the relevant columns
op_col = 'operation_evi'
cat_col = 'category_evi'
mem_col = 'memory'

# Set up the mediation model for each operation
for operation in ['maintain', 'replace', 'suppress']:

    # Load the data
    data = pd.read_csv(f'bootstrap_{space}_study_{condition}_iteration_dataframe.csv')
    
    # Filter the data for the current operation
    operation_data = data[data['operation'] == operation]
    
    # Set up the mediation model
    try:
        m = Mediation(operation_data['operation_evi'], operation_data['memory'], operation_data['category_evi'])
    except ModelSpecificationError as e:
        print(f"Error for operation {operation}: {e}")
        continue
    
    # Run the mediation analysis
    results = m.compute(alpha=.05, n_boot=1000)
    
    # Print the results
    print(f"Results for operation {operation}:")
    print(f"  Total effect: {results.total}")
    print(f"  Direct effect: {results.direct}")
    print(f"  Indirect effect: {results.indirect}")
    print(f"  p-value: {results.p:.3f}")