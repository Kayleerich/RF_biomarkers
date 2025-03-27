import pandas as pd 
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, classification_report
from sklearn.metrics import f1_score as f1
from sklearn.model_selection import train_test_split
from fgclustering import FgClustering
from fgclustering.statistics import calculate_local_feature_importance
import fgclustering.utils as utils

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

class RFBiomarkers():
    def __init__(self, data, predictors, targets_col, toi, outdir, fileID, 
                 write, min_thresh=None, max_thresh=None, RF_type='classifier'):
        self.data = data ## dataframe
        self.predictors = predictors ## columns to use as predictors
        self.targets = targets_col ## column to use as target
        self.toi = toi ## specific target of interest
        self.max_thresh = max_thresh
        self.RF_type = RF_type
        self.outdir = outdir
        self.fileID = fileID
        self.write = write
        self.info_file = Path(f'{self.outdir}/{self.fileID}info.txt')
        self.params_file = Path(f'{self.outdir}/{self.fileID}parameters.tsv')

        self.y = self.data[self.targets]
        [self.level1, self.level2] = list(set(self.y))
        self.X = self.data.drop(
            columns=[c for c  in self.data.columns.to_list() if c not in self.predictors])
        self.X = self.X.T.drop_duplicates().T
        if not min_thresh:
            min_thresh=(round(self.X.shape[0]*0.05))
        self.min_thresh = min_thresh
        if not max_thresh:
            max_thresh=(round(self.X.shape[0]*0.95))
        self.max_thresh = max_thresh
        while True: ## can only work if all predictor values are 1 or 0
            prev_shape = self.X.shape
            cols_sum = self.X.sum(axis=0, numeric_only=True)
            drop_cols = cols_sum[cols_sum < self.min_thresh].index
            self.X.drop(columns=drop_cols, inplace=True)
            drop_cols = cols_sum[cols_sum > self.max_thresh].index
            self.X.drop(columns=drop_cols, inplace=True)
            if self.X.shape == prev_shape:
                break
        self.X[self.targets] = self.y

    def generate_RF(self, best_seeds=True, seeds=(None, None), 
                    train=True, test_size=0.2, plot=True, n=50):
        """
        Generate sample clusters from random forest classifier and ranks features by importance for predicting correct cluster.
        plot: bool, plot feature importance for top n features (default=False)
        n: int, number of features to plot (default=50)
        Adds attributes to class object: 
            rf: RandomForestClassifier object
            seeds: tuple (i, j) with random state values for selecting training data (i) and running model (j)
            X_train, X_test, y_train, y_test: pandas DataFrames containing training and test data 
            y_pred: numpy array of predicted values
        """
        if self.RF_type == 'regressor':
            if self.write != 'none':
                with open(self.info_file, "a") as f:
                    f.write('Regression model not implemented yet, sorry!\n')
            raise NotImplementedError('NotImplementedError: Regression model not implemented yet, sorry!')
        elif self.RF_type == 'classifier':
                if self.y.apply(isinstance, args = [float]).any(): 
                    if len(set([x for x in self.y])) > 10:
                        print('Warning: target values are numeric with more than 10 categorical values. Consider using regression model (not implemented yet, sorry!)')
                        if self.write != 'none':
                            with open(self.info_file, "a") as f:
                                f.write('Warning: target values are numeric with more than 10 categorical values. Consider using regression model (not implemented yet, sorry!)\n')

        
        def _get_best_seeds(range1=10, range2=10): 
            max_accuracy=(0, 0, 0, 0, 0)
            for i in range(range1): 
                X_train, X_test, y_train, y_test = train_test_split(self.X.drop(self.targets, axis=1, inplace=False), 
                                                                    self.y, stratify=self.y, test_size=test_size, random_state=i)
                for j in range(range2):
                    rf = RandomForestClassifier(random_state=j)
                    rf.fit(X_train, y_train)
                    y_pred = rf.predict(X_test)
                    if accuracy_score(y_test, y_pred) >= max_accuracy[0]:
                        if f1(y_test, y_pred, pos_label=self.toi) >= max_accuracy[1]:
                            if precision_score(y_test, y_pred, pos_label= self.toi) >= max_accuracy[2]:
                                max_accuracy = (accuracy_score(y_test, y_pred), 
                                                f1(y_test, y_pred, pos_label=self.toi), 
                                                precision_score(y_test, y_pred, pos_label=self.toi), 
                                                i, j)
                                if self.write == 'all':
                                    info_str = [f'\ntraining seed: {i}; RF seed: {j}\n', 
                                                classification_report(y_test, y_pred)]
                                    with open(self.info_file, "a") as f:
                                        f.writelines(info_str)
            return max_accuracy
        
        if best_seeds:
            if self.write != 'none':
                with open(self.info_file, "a") as f:
                    f.write('\nLooking for best seeds...')
            max_accuracy = _get_best_seeds()
            self.seeds = (max_accuracy[3], max_accuracy[4])
        else: 
            self.seeds = seeds
        if not train:
            self.seeds[0] = None

        info_str = [f"\nSeed used during training data selection (dataselect_seed): {self.seeds[0]}\n", 
                    f"Seed used for RF random state (model_seed): {self.seeds[1]}\n", 
                    f'Train model (train): {train}\n']
        if self.write != 'none':
            with open(self.info_file, "a") as f:
                f.writelines(info_str)
        with open(self.params_file, "a") as p:
            p.writelines(f"dataselect_seed\t{self.seeds[0]}\n")
            p.writelines(f"model_seed\t{self.seeds[1]}\n")
            p.writelines(f"train\t{train}\n")
            

        if train:
            if self.write != 'none':
                with open(self.info_file, "a") as f:
                    f.write(f'Splitting data: {round((1-test_size)*100)}% training, {round((test_size)*100)}% testing (test_size)... \n')
            with open(self.params_file, "a") as p:
                p.writelines(f"test_size\t{test_size}\n")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X.drop(self.targets, axis=1, inplace=False), 
                self.y, stratify=self.y, test_size=test_size, random_state=self.seeds[0])
            if self.RF_type == 'classifier':
                if self.write != 'none':
                    with open(self.info_file, "a") as f:
                        f.write(f'\nGenerating random forest classifier...\n')
                self.rf = RandomForestClassifier(random_state=self.seeds[1])
            elif self.RF_type == 'regressor':
                if self.write != 'none':
                    with open(self.info_file, "a") as f:
                        f.write(f'\nGenerating random forest regressor...\n')
                self.rf = RandomForestRegressor(random_state=self.seeds[1])
            if self.write != 'none':
                with open(self.info_file, "a") as f:
                    f.write(f'\nFitting model...\n')
            self.rf.fit(self.X_train, self.y_train)
            if self.write != 'none':
                with open(self.info_file, "a") as f:
                    f.write(f'\nTesting model...\n')
            self.y_pred = self.rf.predict(self.X_test)
        
        else: 
            with open(self.params_file, "a") as p:
                p.writelines(f"test_size\tNone\n")
            self.rf = RandomForestClassifier(random_state=self.seeds[1]) 

        if plot:
            plt.figure(figsize=(15,15))
            sns.set_theme(font_scale=0.8)
            feature_importances = pd.Series(self.rf.feature_importances_, 
                                            index=self.X_train.columns
                                            ).sort_values(ascending=False)
            feature_importances[0:n].plot.bar() 
            plt.suptitle(
                f"Important features before RF clustering",
                fontsize=20)
            plt.savefig(Path(f'{self.outdir}/{self.fileID}important_feat_noclust.png'))
            plt.savefig(Path(f'{self.outdir}/{self.fileID}important_feat_noclust.pdf'))  

        return
    

    def plot_RFtrees(self, n=6):
        """
        Generate plots of decision trees from random forest
            n: int, number of plots to generate (default=6)
        """        
        from sklearn import tree
        fn=self.X.drop(self.targets, axis=1, inplace=False).columns.to_list()
        cn=self.y.unique()
        for i in range(n):
            fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
            tree.plot_tree(self.rf.estimators_[i],
                        feature_names = fn, 
                        class_names=cn,
                        filled = True)
            plt.savefig(Path(f'{self.outdir}/{self.fileID}dectree_{i}.png'))
            plt.savefig(Path(f'{self.outdir}/{self.fileID}dectree_{i}.pdf'))   
        return

    
    def generate_RFclusters(self, plot=True, n=100):
        """
        Generate sample clusters from random forest classifier and ranks features 
        by importance for predicting correct cluster.
            plot: bool, plot feature importance for top n features (default=False)
            n: int, number of features to plot (default=100)
        Adds attributes to class object: 
            fgc: FgClustering object
            importance_local: pandas DataFrame with Importance score for each feature for each cluster 
        """
        if self.write != 'none':
            with open(self.info_file, "a") as f:
                f.write(f'\nClustering random forest...\n')
        self.fgc = FgClustering(model=self.rf, data=self.X, 
                                target_column=self.targets, random_state=self.seeds[1])
        self.fgc.run(k=2) 
        self.fgc.calculate_statistics(data=self.X, target_column=self.targets)
        calculate_local_feature_importance(self.fgc.data_clustering_ranked, 1000) ## p-values
        importance_dict = {"Feature": self.fgc.p_value_of_features_per_cluster.index}
        for cluster in self.fgc.p_value_of_features_per_cluster.columns: 
            importance_dict[f"cluster{cluster}_importance"] = utils.log_transform(
                self.fgc.p_value_of_features_per_cluster[cluster].to_list())
        self.importance_local = pd.DataFrame(importance_dict)
        if plot: 
            self.fgc.plot_feature_importance(thr_pvalue=0.01, top_n=n, num_cols=5, 
                                             save=Path(f'{self.outdir}/{self.fileID}cluster'))
        return 
    

    def get_biomarkers(self, min_target=None):
        """
        Identify potential orthogroups of interest
            min_target: int, minimum number of target samples with orthogroup present in the cluster of interest
        Adds attributes to class object: 
            min_target: minimum number of targets in cluster (default is 80%)
            array_dict: dictionary of distributions for each orthogroup 
            clust_ogs: list of orthogroups of interest in cluster of interest
        """
        if self.write != 'none':
            with open(self.info_file, "a") as f:
                f.write(f'\nIdentifying Biomarkers...\n')

        all_ranked_features = self.fgc.data_clustering_ranked.drop(columns=["cluster", "target"], 
                                                                   axis=1, inplace=False).columns.to_list()
        self.array_dict = {}
        clust_ogs = [] 
        clust_comp = self.fgc.data_clustering_ranked[["cluster", "target"]].groupby(
            ["cluster", "target"], as_index=False).size() ## all cluster sizes 
        
        clust_dict = clust_comp.to_dict('index')
        clust = max((int(d['size']), d['cluster']) for d in clust_dict.values() if d['target'] == self.toi) ## cluster of interest size and ID 
        clust_size = clust_comp[['cluster', 'size']].groupby('cluster', as_index=False).sum()
        if not min_target:
            self.min_target = round(clust[0]*0.8)
        else:
            self.min_target = round(min_target)

        info_str = [f"Cluster of interest (coi): {clust[1]}\n", 
               f"Cluster size (coi_size): {clust_size[clust_size['cluster'] == clust[1]]['size'].iat[0]}\n", 
               f"Number of targets in cluster (num_toi): {clust[0]}\n", 
               f"Minimum number of targets in cluster (min_target): {self.min_target}\n"]
        if self.write != 'none':
            with open(self.info_file, "a") as f:
                f.writelines(info_str)
        with open(self.params_file, "a") as p:
            p.writelines(f"coi\t{clust[1]}\n")
            p.writelines(f"coi_size\t{clust_size[clust_size['cluster'] == clust[1]]['size'].iat[0]}\n")
            p.writelines(f"num_toi\t{clust[0]}\n")
            p.writelines(f"min_target\t{self.min_target}\n")
            
        if (len(set(self.fgc.data_clustering_ranked['target'].to_list())) == len(self.y.unique())): ## both should be equal to 2
            for og in all_ranked_features: 
                og_df = self.fgc.data_clustering_ranked[['cluster', 'target', og]]
                o_cnt = og_df.groupby(og_df.columns.tolist(),as_index=False).size()
                o_cnt.rename({'cluster': 'c', 'target': 't', og: og, 'size': 's'}, axis=1, inplace=True)
                self.array_dict[og] = o_cnt['s'].to_list()
                tcp = o_cnt[(o_cnt['t'] == self.toi) & (o_cnt['c'] == clust[1]) & (o_cnt[og] == 1)]['s'].iat[0] # target, cluster of interest, present
                ncp = o_cnt[(o_cnt['t'] != self.toi) & (o_cnt['c'] == clust[1]) & (o_cnt[og] == 1)]['s'].iat[0] # non-target, cluster of interest, present
                tnp = o_cnt[(o_cnt['t'] == self.toi) & (o_cnt['c'] != clust[1]) & (o_cnt[og] == 1)]['s'].iat[0] # target, not cluster of interest, present
                nnp = o_cnt[(o_cnt['t'] != self.toi) & (o_cnt['c'] != clust[1]) & (o_cnt[og] == 1)]['s'].iat[0] # non-target, not cluster of interest, present
                tca = o_cnt[(o_cnt['t'] == self.toi) & (o_cnt['c'] == clust[1]) & (o_cnt[og] != 1)]['s'].iat[0] # target, cluster of interest, absent
                nca = o_cnt[(o_cnt['t'] != self.toi) & (o_cnt['c'] == clust[1]) & (o_cnt[og] != 1)]['s'].iat[0] # non-target, cluster of interest, absent
                tna = o_cnt[(o_cnt['t'] == self.toi) & (o_cnt['c'] != clust[1]) & (o_cnt[og] != 1)]['s'].iat[0] # target, not cluster of interest, absent
                nna = o_cnt[(o_cnt['t'] != self.toi) & (o_cnt['c'] != clust[1]) & (o_cnt[og] != 1)]['s'].iat[0] # non-target, not cluster of interest, absent
                ## check if Target presence in Cluster is greater than nontarget presence in either cluster
                ## check if overall presence in Cluster is greater than absence in Cluster 
                ## check if Target presence in Cluster is greater than Target presence in noncluster
                ## check minimum threshold for Target presence in Cluster 
                if (tcp > (ncp + nnp)) & ((tcp + ncp) > (tca + nca)) & (tcp > tnp) & ((nna + tna) > (nnp + tnp)) & (tcp >= self.min_target): 
                    clust_ogs.append(og)
            self.clust_ogs = clust_ogs 
            importance_file= Path(f'{self.outdir}/{self.fileID}feature_importance.tsv')
            pval_file = Path(f'{self.outdir}/{self.fileID}feature_distribution_pvalues.tsv')
            feat_file = Path(f'{self.outdir}/{self.fileID}important_features_list.txt')
            
            with open(importance_file, "w") as impfile:
                self.importance_local.sort_values(
                    by=f"cluster{clust[1]}_importance", ascending=False).to_csv(
                        impfile, sep='\t', index=False)
            with open(pval_file, "w") as pvalfile:
                pval_df = self.fgc.p_value_of_features_per_cluster
                pval_df.columns = ['p-value_1', 'p-value_2']
                distr_df = pd.DataFrame(self.array_dict, index=[f'{self.level1}_1_absent', 
                                                                f'{self.level1}_1_present', 
                                                                f'{self.level2}_1_absent', 
                                                                f'{self.level2}_1_present', 
                                                                f'{self.level1}_2_absent', 
                                                                f'{self.level1}_2_present', 
                                                                f'{self.level2}_2_absent', 
                                                                f'{self.level2}_2_present'])
                pd.concat([distr_df, pval_df.T]).to_csv(
                        pvalfile, sep='\t', index=True)
            with open(feat_file, "w") as featfile:
                for og in self.clust_ogs:
                    featfile.write(f'{og}\n')

            info_str = [f'Number of potential biomarkers (num_biomarker): {len(self.clust_ogs)}\n', 
                        f'Feature importance scores written to: {str(importance_file.resolve())}\n', 
                        f'Feature distributions and p-values written to: {str(pval_file.resolve())}\n']
            if self.write != 'none':
                with open(self.info_file, "a") as f:
                    f.writelines(info_str)
            with open(self.params_file, "a") as p:
                p.writelines(f'num_biomarker\t{len(self.clust_ogs)}\n')
            if self.write == 'all':
                with open(self.info_file, "a") as f:
                    f.write(f'List of all orthogroups of interest (list_biomarker): {", ".join(self.clust_ogs)}\n')
                with open(self.params_file, "a") as p:
                    p.writelines(f'list_biomarker\t{self.clust_ogs}\n')
        return
                
    def identical_distr(self):
        """
        Identify orthogroups of interest with identical distribution across clusters
        Adds attributes to class object: 
            identical_features_dict: dictionary of lists containing features with 
                                     high importance and identical distributions
        """
        self.identical_features_dict = {}
        ident_lst = self.data[self.predictors].T.reset_index().groupby(
            list(self.data[self.predictors].T))['index'].agg(list).tolist()
        for og in self.clust_ogs:
            for lst in ident_lst:
                if og in lst:
                    self.identical_features_dict[og] = lst
        return


    def distribution_subplots(self, colors=None, stacked=True):
        """
        Plot distribution graphs for features of interest
        """
        if not colors:
            colors = ["#FDF7EE", "#6F1010"]
        hatches = ['', '']
        edge=None
        leg_dict = {'Orthologue present': ["#FDF7EE", ''], 
                    'Orthologue absent': ["#6F1010", '']}
        if stacked:
            edge='grey'
            hatches = ['', 'xx']
            leg_dict['High SCC isolate'] = ["white", '']
            leg_dict['Low SCC isolate'] = ["white", 'xx']
        num_cols = 6 if len(self.clust_ogs) >= 6 else len(self.clust_ogs)+1
        num_rows = int(np.ceil(len(self.clust_ogs) / num_cols))
        plt.figure(figsize=(num_cols * 7, num_rows * 7))
        plt.subplots_adjust(top=0.95, hspace=0.8, wspace=0.8)

        for i, og in enumerate(self.identical_features_dict.keys()):
            df = pd.DataFrame([self.array_dict[og][0:2], self.array_dict[og][2:4], 
                               self.array_dict[og][4:6], self.array_dict[og][6:]],
                            index=["HIGH_1", "LOW_1", "HIGH_2", "LOW_2"],
                            columns=["absent", "present"])
            df['orthogroup'] = og 
            df.reset_index(inplace=True)
            df[['SCC_level', 'cluster']] = df['index'].str.split('_', expand=True)
            df.drop(columns='index', inplace=True)
            
            df_melt = pd.melt(df, id_vars=['orthogroup', 'SCC_level', 'cluster'], 
                              var_name='presence') 
            df_melt['cluster'] = 'Cluster ' + df_melt['cluster'].astype(str)
            df_melt.set_index(["presence", "cluster", "SCC_level"], inplace=True)
            df_melt["vcs"] = df_melt.groupby(level=["presence", "cluster"]).cumsum(numeric_only=True)
            df_melt.drop(columns='orthogroup', inplace=True)

            for en, g in enumerate(df_melt.groupby("SCC_level")):
                ax = plt.subplot(num_rows, num_cols, i + 1)
                sns.barplot(data=g[1],
                            x="cluster",
                            y="vcs",
                            hue="presence",
                            hatch=hatches[en],
                            palette=colors,
                            zorder=-en, # so first bars stay on top
                            edgecolor=edge, 
                            ax=ax)
            
            ax.legend_.remove() 
            if len(self.identical_features_dict[og]) == 1:
                others = f''
            elif len(self.identical_features_dict[og]) == 2:
                others = f'and {self.identical_features_dict[og][1]}'
            else:
                others = f'+ {len(self.identical_features_dict[og])-1} others'
            ax.set_title(f"{og} {others}",fontdict={'fontsize': 25})
            ax.set_ylabel(' ')
            ax.set_xlabel(' ')
            for bar in ax.patches:
                bar.set_linewidth(2)
                bar.set_edgecolor('grey')
            ax.set_axisbelow(True)
            ax.yaxis.grid(color='gray', linestyle='dashed')

        leg_artists = []
        leg_labels = []
        for k in leg_dict.keys():
            p = Patch(facecolor=leg_dict[k][0], 
                      hatch=leg_dict[k][1], 
                      edgecolor=edge)
            leg_artists.append(p)
            leg_labels.append(k)

        if len(self.clust_ogs) < num_rows*num_cols:
            ax = plt.subplot(num_rows, num_cols, i+2)
            ax.axis("off")
            leg = ax.legend(leg_artists, leg_labels, loc='center left', 
                            labelspacing=1, handleheight=2.5, handlelength=2) 
        else:
            leg = ax.legend(leg_artists, leg_labels, loc='upper left', 
                            labelspacing=1, handleheight=2.5, handlelength=2) 
        for patch in leg.get_patches():
            patch.set_y(0)
        
        plt.setp(ax.get_legend().get_texts(), fontsize='20')      
        plt.setp(ax.get_legend().get_title(), fontsize='20')  
        plt.tight_layout(pad=2)
        plt.savefig(Path(f'{self.outdir}/{self.fileID}distributions.png'))
        plt.savefig(Path(f'{self.outdir}/{self.fileID}distributions.pdf'))   
        return       