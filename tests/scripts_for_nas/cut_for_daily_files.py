from datetime import datetime

def create_daily_files(input_file: str, path_output: str) -> None:
    
    # Read the input file
    with open(input_file, 'r') as f:
        # Read the header lines
        header              = f.readline().strip()
        header_labels       = f.readline().strip()
        header_units        = f.readline().strip()
        header_labels_short = f.readline().strip()

        current_day = None
        current_file = None

        # Process each line in the input file
        for line in f:
            parts = line.strip().split(',')
            timestamp_str = parts[0].strip('"')
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            day = timestamp.strftime('%Y%m%d')

            # Check if a new day has started
            if day != current_day:
                # Close the current file (if it's open)
                if current_file:
                    current_file.close()

                # Create a new output file for the current day
                output_file = f"{path_output}meteo_{day}.dat"
                current_file = open(output_file, 'w')

                # Write the header lines to the new file
                current_file.write(f"{header}\n")
                current_file.write(f"{header_labels}\n")
                current_file.write(f"{header_units}\n")
                current_file.write(f"{header_labels_short}\n")

                current_day = day

            # Write the current line to the appropriate daily file
            current_file.write(line)

        # Close the last daily file (if it's open)
        if current_file:
            current_file.close()

if __name__ == "__main__":
    input_file  = "../test_for_nas/meteo/meteo_20230724.dat"  # Replace with the actual name of your input file
    path_output = "../test_for_nas/meteo/data/"
    create_daily_files(input_file, path_output)
