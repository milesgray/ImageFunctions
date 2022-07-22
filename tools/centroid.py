import gc

import torch
import torch.nn as nn

import numpy as np

import umap
# TODO: import TSNE
# TODO: import fast_tsne

import tqdm.tqdm as tqdm

import matplotlib.pyplot as plt

class CentroidDistanceConcentrationPlot:
    def __init__(self, loader, loss_temp=0.1, logger=print, experiment=None):
        self.epoch = 0        
        self.loss_temp = loss_temp
        self.logger = logger
        self.experiment = experiment
        self.loader = loader
        self.dataset_size = len(self.loader.dataset.mapping)
        self.batch_size = self.loader.dataset.batch_size
        self.features = None
        self.labels = None
        self.classes = self.loader.dataset.classes
        self.num_class = self.loader.dataset.num_class
        self.reduced_features = None
        self.centroids = None
        self.distances = None
        self.concentrations = None

    def build(self, epoch, net, loader=None, 
              mode="tsne", draw=True, save=True, verbose=True):
        if loader: self.loader = loader
        self.features, self.labels = self._evaluate(net)

        if draw:
            if mode == "tsne":
                try:
                    self.reduced_features = self._transform_fast_tsne(self.features.detach().cpu().numpy(), verbose=verbose)
                except Exception as e:
                    self.logger(f"[ERROR]\tFast TSNE failed:\n{e}")
                    self.reduced_features = self._transform_tsne(self.features.detach().cpu().numpy(), verbose=verbose)
            elif mode == "umap":
                self.reduced_features = self._transform_umap(self.features.detach().cpu().numpy(), verbose=verbose)
            elif mode == "svd":
                self.reduced_features = torch.svd(self.features)
            self.reduced_features = torch.Tensor(self.reduced_features).cuda()
            centroid_results = self._find_centroids(self.reduced_features, self.labels, verbose=verbose)
            self.centroids, self.distances, self.concentrations = centroid_results
            self.draw(epoch, self.reduced_features, 
                      self.labels, self.centroids, self.concentrations, 
                      figname=f"{mode} Clusters", 
                      save=save, verbose=verbose)
        
        return self._find_centroids(self.features, self.labels, verbose=verbose)

    def draw(self, epoch, features, labels, centroids, concentrations, save=True, 
             figsize=20, figname='Clusers', folder=args.results_dir,
             center_cmap='jet', edge_cmap='cividis', verbose=True):
        features = features.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        centroids = centroids.detach().cpu().numpy()
        concentrations = concentrations.detach().cpu().numpy()

        plt.rcParams["figure.figsize"] = (figsize, figsize)
        fig, ax = plt.subplots()
        num_labels = len(self.classes)
        center_colors = plt.get_cmap(center_cmap)(np.linspace(0, 1, num_labels))
        edge_colors = plt.get_cmap(edge_cmap)(np.linspace(0, 1, num_labels))
        ax.scatter(features[:, 0], features[:, 1], 
                    marker='.', 
                    alpha=0.75, 
                    c=labels,
                    cmap=center_cmap,
                    edgecolors=edge_colors)
        ax.scatter(centroids[:, 0], centroids[:, 1], 
                marker='8', 
                alpha=0.7, 
                s=10 * figsize,
                color=center_colors,                
                edgecolors="#FFFFFF")
        ax.scatter(centroids[:, 0], centroids[:, 1], 
                marker='o', 
                alpha=0.001, 
                s=np.pi * figsize ** (np.exp(1 + concentrations) / 2),
                color=center_colors,                
                edgecolors="#FFFFFF")
        ax.scatter(centroids[:, 0], centroids[:, 1], 
                marker='o', 
                alpha=1, 
                s=np.pi * figsize ** (np.exp(1 + concentrations) / 2),
                facecolors="none",
                linewidth=0.5,
                edgecolors=edge_colors)
        
        plt.title(f'{figname} {epoch}')

        if save:
            plt.savefig(f'{folder}/{figname}-{epoch:05d}.png')
            if self.experiment: self.experiment.log_image(f'{folder}/{figname}-{epoch:05d}.png')           
            if verbose: self.logger(f"[INFO]\t Saved as {folder}/{figname}-{epoch:05d}.png")
        else:
            if self.experiment: self.experiment.log_figure()
            plt.show()

        plt.close(fig)
        gc.collect()

    def _transform_tsne(self, features, iters=1000, perplexity=20, n_components=2, init="pca", 
                    method="barnes_hut", n_jobs=-1, verbose=True):
        try:
            if verbose: self.logger(f"[INFO]\tStarting T-SNE fit transform on {features.shape} to {(features.shape[0], n_components)}, training for {iters} iterations")
            start = time.time()
            tsne_embed = TSNE(perplexity=perplexity, n_components=n_components, init="pca", n_iter=iters, method=method, n_jobs=n_jobs).fit_transform(features)
            if verbose: self.logger(f"[INFO]\tT-SNE took {(time.time()-start)/60} minutes")
            return tsne_embed
        except Exception as e:
            self.logger(f"[ERROR]\tFailed T-SNE: {e}")
            return features

    def _transform_fast_tsne(self, features, verbose=True):
        try:
            if verbose: self.logger(f"[INFO]\tStarting Fast T-SNE fit transform on {features.shape} to {(features.shape[0], 2)}")
            start = time.time()
            tsne_embed = fast_tsne(features)
            if verbose: self.logger(f"[INFO]\tFast T-SNE took {(time.time()-start)/60} minutes")
            return tsne_embed
        except Exception as e:
            self.logger(f"[ERROR]\tFailed Fast T-SNE: {e}")
            return features

    def _transform_umap(self, features, verbose=True):
        mapper = umap.UMAP(random_state=42).fit(features)
        return mapper.embedding_.T

    def _find_centroids(self, features, labels, attr_name="image", dist_name="Projection", msg_break="\t", verbose=True):        
        num_label = len(self.classes)
        if verbose: self.logger(f"[DEBUG]\t{num_label} unique class categories")
        centroids = torch.empty((num_label, features.shape[1])).to(features.device)
        distances = torch.empty((num_label,)).to(features.device)
        concentrations = torch.empty((num_label,)).to(features.device)
        message = f"[INFO]\tAvg Distance to each {attr_name}'s {features.shape[1]}D {dist_name} centroid:{msg_break}"
        for label_index in tqdm(range(num_label), desc="Finding Centroids"):
            bool_index = (labels == label_index).squeeze()
            centroids[label_index] = self._find_centroid(features[bool_index, :])
            distances[label_index], concentrations[label_index] = self._get_distance_stats(features[bool_index, :], centroids[label_index])
            
        # https://arxiv.org/pdf/2005.04966v1.pdf - We normalize φ for each set of prototypes Cm such that they have a mean of τ 
        concentrations = self._normalizer(concentrations, self.loss_temp)
        for label_index in range(num_label):
            message = f"{message}{label_index}: {distances[label_index]:.4f}({concentrations[label_index]:.4f}){msg_break}"
        
        correct, closest = self._eval_closest_centroid(features, centroids, labels)
        total_correct = correct.sum()
        percent_correct = total_correct / correct.shape[0] * 100

        if self.experiment: self.experiment.log_metric("cluster_total_correct", total_correct)
        if self.experiment: self.experiment.log_metric("cluster_percent_correct", percent_correct)

        message = f"{message}\n\t]]~~~> > >-{attr_name.upper()}-< < <~~~[\t{total_correct} Correctly placed near centroid @ {percent_correct:.2f}%] \n"

        if verbose: self.logger(message)    
        
        return centroids, distances, concentrations    

    def _eval_closest_centroid(self, features, centroids, labels, dists=None): 
        if dists is None:
            dists = torch.cdist(features, centroids)   
        closest = torch.argmin(dists, axis=1).to(features.device)
        correct = torch.eq(closest, torch.squeeze(labels.to(features.device))).float()
        return correct, closest    

    def _find_centroid(self, features):
        length = features.shape[0]
        dims = features.shape[1]
        result = torch.empty((dims,))
        for i in range(dims):
            result[i] = torch.sum(features[:, i]) / length
        return result

    def _get_distance_stats(self, features, centroid, a=10, formula=2, dist=None):
        if dist is None:
            dist = torch.cdist(features, torch.unsqueeze(centroid, 0))
        #φ = SUM |Vz − c|2
        #     Z log(Z + α)
        if formula == 1: 
            density = dist.sum() / (features.shape[0] * math.log(features.shape[0] + a)) 
        elif formula == 2:
            density = (dist ** 0.5).mean() / (math.log(dist.shape[0] + a)) 
        
        return dist.mean(), density

    def _normalizer(self, density, temp, formula=2):
        try:
            if formula==1:
                norm = values / torch.linalg.norm(values)
            elif formula == 2:
                density = density.clip(torch.percentile(density,10),torch.percentile(density,90)) #clamp extreme values for stability
                norm = temp*density/density.mean()  #scale the mean to temperature 
            return norm
        except:
            return density

    def _evaluate(self, net):
        classes = []
        features = []
        with torch.no_grad():
            with tqdm(self.loader, desc='Feature extracting', total=self.dataset_size//self.batch_size) as bar:
                for i, batch in enumerate(bar):
                    feature = net(batch["images"][0].cuda(non_blocking=True))                    
                    features.append(feature)
                    classes.append(batch["label"])
                    bar.set_postfix({"Feature mean": feature.mean()})
                    if i >= self.dataset_size//self.batch_size:
                        break
            # [D, N]
            features = torch.cat(features)
            classes = torch.cat(classes)        
        return features, classes
    
    def make_batch(self, centroids, targets):    
        return np.take(centroids, 
                    targets, 
                    axis=0)