import os
import pickle

folder_path = 'dataset/Tongji'

pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]

for file_name in pkl_files:
    file_path = os.path.join(folder_path, file_name)
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        cycle_data = data.get('cycle_data', [])
        cycle_numbers = [entry['cycle_number'] for entry in cycle_data if 'cycle_number' in entry]

        if cycle_numbers:
            print(f"File: {file_name}")
            print(f"Min cycle_number: {min(cycle_numbers)}")
            print(f"Max cycle_number: {max(cycle_numbers)}\n")
        else:
            print(f"File: {file_name} has no cycle_number data.\n")
    except Exception as e:
        print(f"Failed to process {file_name}: {e}\n")
