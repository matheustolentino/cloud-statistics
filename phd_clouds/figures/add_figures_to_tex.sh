#!/bin/bash

# Check if correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <path_to_figures> <path_to_tex_file>"
    exit 1
fi

# Store the paths provided as arguments
figures_path="$1"
tex_file="$2"

# Check if the provided paths are valid
if [ ! -d "$figures_path" ]; then
    echo "Error: '$figures_path' is not a valid directory."
    exit 1
fi

# Create a new .tex file with article document type if it doesn't exist
if [ ! -f "$tex_file" ]; then
    echo "\documentclass{article}" > "$tex_file"
    echo "\usepackage[letterpaper,top=2cm,bottom=2cm,left=3cm,right=3cm,marginparwidth=1.75cm]{geometry}" >> "$tex_file"
    echo "\usepackage{amsmath}" >> "$tex_file"
    echo "\usepackage{graphicx}" >> "$tex_file"
    echo "\begin{document}" >> "$tex_file"
    echo "Created $tex_file with article document type."
fi

# Loop through each PNG file in the figures directory
for figure_file in "$figures_path"/*.png; do
    # Extract only the filename without the path
    filename=$(basename "$figure_file")

    # Add the figure to the tex file
    echo "\begin{figure}[htbp]" >> "$tex_file"
    echo "\centering" >> "$tex_file"
    echo "    \includegraphics[width=1.\textwidth]{$filename}" >> "$tex_file"
    echo "\end{figure}" >> "$tex_file"

    echo "Added $filename to $tex_file"
done

# Add \end{document} to properly close the document
echo "\end{document}" >> "$tex_file"

echo "All figures added to $tex_file successfully."

