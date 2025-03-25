import argparse
from pathlib import Path
import sys
import traceback
from time import localtime, strftime
import pandas as pd 
from rfbiomarker import RFBiomarkers

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='%(prog)s [-h] -i INPUT -o DIR -c COLUMN -t TARGET [-d -f -r -p -v -w --min --max --test_size --seeds --force --unsup]', description='Uses random forest classifier and clustering to identify biomarkers')
    parser.add_argument('-i', '--input', metavar='', type=argparse.FileType('r'), default=sys.stdin, help='Input TSV with column names (default from stdin)')
    parser.add_argument('-o', '--outdir', metavar='', type=str, required=False, help='Directory to save output files (default is current directory)') 
    parser.add_argument('-c', '--targets_col', metavar='', type=str, required=True, help='Name of column containing target values')
    parser.add_argument('-d', '--ID_col', metavar='', type=str, required=False, help='Name of column containing sample IDs')    
    parser.add_argument('-t', '--toi', metavar='', type=str, required=False, help='Target value of interest')
    parser.add_argument('-f', '--fileid', metavar='', type=str, required=False, help='Optional name for output files')
    parser.add_argument('-p', '--predictors', metavar='', nargs='+', required=False, help='List of columns to use as predictors (space delim, by default uses all columns except for specified target column and sample IDs column)')
    parser.add_argument('-r', '--remove', metavar='', nargs='+', required=False, help='List of columns to not use as predictors (space delim, opposite of --predictors, i.e. will use all columns in data except for those specified and the target/sample ID columns)')
    parser.add_argument('--min', metavar='', type=int, required=False, help='Minimum number of samples a feature must be present in (default is 5%% of total)')
    parser.add_argument('--max', metavar='', type=int, required=False, help='Maximum number of samples a feature must be present in (default is 95%% of total)')
    parser.add_argument('--test_size', metavar='', type=float, default=0.2, help='Test size used to train model (default test size is 0.2, i.e. will use 80%% of data to train model and 20%% to test)')
    parser.add_argument('--seeds', metavar='', type=int, nargs=2, required=False, help='Seed/random state values to use for subsampling training data and running model (by default will calculate best seeds)')
    parser.add_argument('-w','--write', metavar='', type=str, required=False, choices=['none', 'all'], help='Model data to write to parameters.txt. Options are: "none" or "all" (default behavior writes pertinent information)')
    parser.add_argument('--force', action='store_true', required=False, help='Overwrite previous output files')
    parser.add_argument('--unsup', action='store_true', required=False, help='Not recommended! Run unsupervised RF (by default will train RF on 80%% of data, use --test_size to change)')
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')
    args = parser.parse_args()
    
    if not args.outdir:
        outdir = Path.cwd()
    elif Path(args.outdir).exists():
        outdir = Path(args.outdir) 
    else:
        try:
            Path(args.outdir).mkdir()
        except PermissionError:
            write_str = f'-o/--outdir: unable to create "{args.outdir}/", permission denied'
            raise PermissionError(write_str)
        outdir = Path(args.outdir)
    [fileID, write_str] = [f'{args.fileid}_', f' (ID name for files (fileID): {args.fileid})'] if args.fileid else ['', '']
    write = args.write if args.write else "default"

    if Path(f'{outdir}/{fileID}parameters.tsv').is_file():
        if not args.force:
            write_str = ''.join([f'Analysis output files in {str(outdir.resolve())} ', '"'.join(['with fileID' , args.fileid, '" ']) if args.fileid else '', 'already exist, use --force to ignore this error and overwrite these files'])
            raise FileExistsError(write_str)

    if write != 'none':
        with open(f'{outdir}/{fileID}info.txt', "w") as f:
            f.write(f'RFbiomarkers version 0.1\nStart time: {strftime("%Y-%m-%d %H:%M:%S", localtime())}\n')
            f.write(f'Writing files to (outdir): {outdir.resolve()}{write_str}\n\n')
    with open(f'{outdir}/{fileID}parameters.tsv', "w") as p:
        p.write(f'# RFbiomarkers version 0.1\n# Start time: {strftime("%Y-%m-%d %H:%M:%S", localtime())}\n')
        p.write(f'outdir\t{outdir.resolve()}\n')
        p.write(f'fileID\t{args.fileid}\n')

    try:
        data = pd.read_csv(args.input, sep='\t')
        if args.ID_col in data.columns:
            data.set_index(args.ID_col, inplace=True)

        if args.targets_col in data.columns:
            targets_col = args.targets_col
        else:
            write_str = f'-c/--targets_col: "{args.targets_col}" is not one of the columns in the given data. Check your spelling and make sure your input data is tab-separated'
            raise ValueError(write_str)
        if (args.toi is None) or (args.toi in data[targets_col].to_list()): 
            toi = args.toi
        else:
            write_str = f'-t/--toi: "{args.toi}" is not one of the values in the target column. Check your spelling'
            raise ValueError(write_str)
        if args.predictors:
            p_miss = []
            for p in args.predictors:
                if p in data.columns:
                    continue
                p_miss.append(p)
            if p_miss:
                write_str = f'-p/--predictors: "{", ".join(p_miss)}" were not found in the column names of the given data'
                raise ValueError(write_str)
            predictors = [c for c in args.predictors if c != targets_col]
        elif args.remove:
            predictors = [c for c in data.columns if c != targets_col and c not in args.remove]
        else:
            predictors = [c for c in data.columns if c != targets_col]
        
        min_thresh = args.min 
        max_thresh = args.max 
        if (0 < args.test_size < 1):
            test_size = args.test_size 
        else:
            write_str = f'--test_size: must be a number between 0 and 1'
            raise ValueError(write_str)
        if args.seeds:
            seeds = tuple(args.seeds)
            best_seeds = False
        else:
            seeds = (None, None)
            best_seeds = True
        if args.unsup:
            train = False
        else:
            train = True

        rfcls = RFBiomarkers(data, 
                            predictors, 
                            targets_col, 
                            toi, 
                            outdir, 
                            fileID, 
                            write, 
                            min_thresh, 
                            max_thresh)
        
        if write != 'none':
            with open(f'{outdir}/{fileID}info.txt', "a") as f:
                f.write(f'Name of target column (targets): {rfcls.__dict__["targets"]}\n')
                f.write(f'Target value of interest (toi): {rfcls.__dict__["toi"]}\n')
                if write == 'all':
                    f.write(f'Predictors/feature column names used (predictors): {", ".join(rfcls.__dict__["predictors"])}\n')
                if [r for r in args.remove if r not in [targets_col, toi, args.ID_col]]:
                    f.write(f'Warning: command line option -r/--remove: none of "{", ".join([r for r in args.remove if r not in [targets_col, toi, args.ID_col]])}" were not found in the column names of the given data and could not be removed as a predictor\n')
                f.write(f'\nMinimum number of samples with feature present needed to be included in model (min_thresh): {rfcls.__dict__["min_thresh"]}\n')
                f.write(f'Maximum number of samples with feature present needed to be included in model (max_thresh): {rfcls.__dict__["max_thresh"]}\n')
        with open(f'{outdir}/{fileID}parameters.tsv', "a") as p:
            p.write(f'targets\t{rfcls.__dict__["targets"]}\n')
            p.write(f'toi\t{rfcls.__dict__["toi"]}\n')
            p.write(f'min_thresh\t{rfcls.__dict__["min_thresh"]}\n')
            p.write(f'max_thresh\t{rfcls.__dict__["max_thresh"]}\n')
            if write == 'all':
                p.write(f'predictors\t{rfcls.__dict__["predictors"]}\n')

        rfcls.generate_RF(best_seeds=best_seeds, 
                        seeds=seeds, 
                        train=train, 
                        test_size=test_size, 
                        plot=False)

        rfcls.generate_RFclusters(plot=False)
        
        rfcls.get_biomarkers() 

        # rfcls.distribution_subplots() # !! Need to add function to class 

        if write != 'none':
            with open(f'{outdir}/{fileID}info.txt', "a") as f:
                f.write(f'\nEnd time: {strftime("%Y-%m-%d %H:%M:%S", localtime())}')
    
    except KeyboardInterrupt:
        write_str = 'Abort by user interrupt. Analysis not finished'
        if write != 'none':
            with open(f'{outdir}/{fileID}info.txt', "a") as f:
                f.write(write_str)
        with open(f'{outdir}/{fileID}parameters.tsv', "a") as p:
            p.write(f'# {write_str}')
        print(write_str)
        sys.exit(1)
    except Exception as exc:
        filename, lineno, funcname, text = traceback.extract_tb(exc.__traceback__)[-1]
        funcname = f'{funcname}():' if funcname != "<module>" else "command line input"
        write_str = f'{type(exc).__name__}: {funcname} {exc}'
        if write != 'none':
            with open(f'{outdir}/{fileID}info.txt', "a") as f:
                f.write(write_str)
        with open(f'{outdir}/{fileID}parameters.tsv', "a") as p:
            p.write(f'# {write_str}\n# WARNING: Analysis not completed')
        print(f'\n{write_str}\n')
        sys.exit(1)
