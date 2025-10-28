# Instruction to use

## Data creation
Into the HnS repository a script named 'run_interactive_test.sh' is provided. Use this script to generate results and copy them into 'input_data/<your_favorite_name\>'.


## Plot generation
Run the bash script 'gen_all_plots.sh input_data/<your_favorite_name\>', this will generate the plots and populate the respective folders. 

**Note**: this could take a while (minutes), since the script to produce the overall plots is quite rought and not optimized.

## Clean and archive plots
The scripts 'clean_plots.sh' and 'archive_plots.sh' can be used to respectivelly cleaning the current plot floders and to archive the current plots into the 'archive/' directory.
