# FuncAnnoClust

The comparative genomics toolkit is implemented as a simple web application for users without programming skills.

The goal of the web application is to automate the analysis of genomic annotations from Rapid Annotations using Subsystems Technology.

It includes reclassification, computation and analysis applying:
- clustering algorithms: BayesianGaussianMixture, KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering, AffinityPropagation
- dimensionality reduction methods: PCA, MDS, t-SNE,
- and statistical tests: PERMANOVA, ANOSIM.

### Install Docker Desktop
https://docs.docker.com/desktop/

### Start docker
```bash
docker build ./ComparativeGenomics-master/ -t test-polars-docker
docker run -p 5000:5000 -d -it test-polars-docker bash
```

### Install packages
It is possible to do through Docker Desktop Terminal.


For version control check FuncAnnoClust-master/environment.yml
```bash 
conda install -c anaconda pandas 
conda install -c anaconda flask
conda install -c anaconda scikit-learn 
conda install -c conda-forge polars
conda install -c conda-forge pyarrow
conda install -c conda-forge scikit-bio
conda install -c plotly plotly
```

### Switch to work mode
In FlaskApp/FlaskApp.py change:
```bash 
if __name__ == '__main__':
    app.run(host='0.0.0.0')
```

### Start web application
```bash 
python3 FlaskApp/FlaskApp.py
```
