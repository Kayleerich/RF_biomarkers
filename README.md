## Identifing potential biomarkers from genomic data using random forest clustering
  
#### Genome assembly: assembly_commands.sh  
Usage for a single strain: `sh assembly_commands.sh strainID`  
Conda environment files: `clust_config/qc.yml`, `clust_config/busco.yml`, `clust_config/bakta.yml`  
<br>
  
#### Clustering and phylogeny: clustering_commands.sh  
Usage for clustering at 90% identity over 80% length: `sh clustering_commands.sh 90 80`  
Conda environment files: `clust_config/ortho.yml`, `clust_config/phylo.yml`  
<br>
  
#### Random Forest and biomarker identification: RF_biomarkers.py  
Basic usage:  
`python RF_biomarkers.py --input file.tsv --outdir path/to/output/directory/ --targets_col target_column --toi target_value [--help]`  

`file.tsv`: tab-separated data from file or stdin with rows containing gene presence or absence (indicated by 1 and 0, respectively) for each sample and genes as column names. Also requires a column containing the target (currently only allows 2 levels).  
Input example:
```
        gene1  gene2  gene3  target_column
        1    0    1    level_1
        1    1    1    level_1
        0    0    1    level_2
        0    0    0    level_2
```
Dependencies: `pandas`, `numpy`, `sklearn`, `fgclustering`, `matplotlib`, `seaborn`

Optional arguments:
```
  -h, --help            show this help message and exit

  -i, --input           Input TSV with column names (default from stdin)

  -o, --outdir          Directory to save output files (default is current directory)

  -c, --targets_col     Name of column containing target values

  -d, --ID_col          Name of column containing sample IDs

  -t, --toi             Target value of interest

  -f, --fileid          Optional name for output files

  -p, --predictors      List of columns to use as predictors (space delim, by default uses all columns
                         except for specified target column and sample IDs column)

  -r, --remove          List of columns to not use as predictors (space delim, opposite of
                        --predictors, i.e. will use all columns in data except for those specified
                        and the target/sample ID columns)

  --min                 Minimum number of samples a feature must be present in (default is 5% of total)

  --max                 Maximum number of samples a feature must be present in (default is 95% of
                        total)

  --test_size           Test size used to train model (default test size is 0.2, i.e. will use 80% of
                        data to train model and 20% to test)

  -s, --seeds           Seed/random state values to use for subsampling training data and running model
                        (by default will calculate best seeds)

  -w, --write           Model data to write to parameters.txt. Options are: "none" or "all" (default
                        behavior writes pertinent information)

  --force               Overwrite previous output files
```
