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
