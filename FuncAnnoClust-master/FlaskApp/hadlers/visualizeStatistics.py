from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering, OPTICS, AffinityPropagation
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
import pandas as pd
import polars as pl
import plotly
import plotly.express as px
import json
import numpy as np

try:
    from skbio.diversity import beta_diversity
    from skbio.stats.distance import anosim, permanova, DistanceMatrix
except ImportError:
    print("!ERROR! Could not import skbio")


def sanitize_distance_matrix(dm):
    """
    Преобразует skbio.DistanceMatrix или numpy-array в чистый 2D numpy.ndarray.
    Убирает NaN, отрицательные артефакты, гарантирует симметрию и ноль на диагонали.
    """
    if hasattr(dm, 'data'):
        data = np.array(dm.data, dtype=np.float64)
    else:
        data = np.array(dm, dtype=np.float64)

    if data.ndim != 2:
        raise ValueError(f"Ожидается 2D матрица расстояний, получено {data.ndim}D")

    # 1. Убираем отрицательные артефакты FP
    np.clip(data, 0, None, out=data)
    # 2. Диагональ = 0, принудительная симметрия
    np.fill_diagonal(data, 0.0)
    data[:] = (data + data.T) / 2.0
    # 3. Заменяем NaN (частая проблема correlation на плоских семплах)
    if np.any(np.isnan(data)):
        max_val = np.nanmax(data) if np.any(~np.isnan(data)) else 1.0
        data[:] = np.nan_to_num(data, nan=max_val)

    return data


def normalize_data(data, method='minmax'):
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'zscore':
        scaler = StandardScaler()
    else:
        raise ValueError("Unsupported normalization method. Choose 'minmax' or 'zscore'.")
    numeric_columns = [col for col in data.columns if col != "Strain"]
    numeric_data = data.select(numeric_columns).to_numpy()
    normalized_data = scaler.fit_transform(numeric_data)
    normalized_df = pl.DataFrame(normalized_data, schema=numeric_columns)
    normalized_df = normalized_df.with_columns(data["Strain"].alias("Strain"))
    return normalized_df


def validate_data_for_clustering(data):
    numeric_columns = [col for col in data.columns if data[col].dtype in (pl.Float64, pl.Int64)]
    if any(data[col].std() == 0 for col in numeric_columns):
        return False, "Some features have zero variance, which may cause issues during clustering."
    return True, "Data is valid for clustering."


def remove_zero_variance_features(data):
    numeric_columns = [col for col in data.columns if data[col].dtype in (pl.Float64, pl.Int64)]
    non_zero_variance_columns = [
        col for col in numeric_columns if data[col].std() > 0
    ]
    return data.select(non_zero_variance_columns + ["Strain"])


def adjust_n_components(n_components, data):
    max_components = min(len(data), n_components)
    return max_components


def clusterization(data, clusterMethods, eps=None, n_clusters="3", linkage='ward', distance_metric='euclidean',
                   random_state=None, tree=None, otu_ids=None, normalization_method='minmax',
                   spectral_affinity='nearest_neighbors', optics_max_eps=None, damping=0.9):
    genes_count = data.getCount()
    distance_matrix = data.getComputedMatrix()

    genes_count_normalized = normalize_data(genes_count, method=normalization_method)
    genes_count_unique = remove_zero_variance_features(genes_count_normalized)

    valid, message = validate_data_for_clustering(
        genes_count_unique.select([col for col in genes_count_unique.columns if col != "Strain"])
    )
    if not valid:
        raise ValueError(message)

    # Precompute distance matrices if necessary
    if distance_metric == 'euclidean' and distance_metric not in distance_matrix:
        distance_matrix['euclidean'] = precomputed_matrix(genes_count_unique, distance_metric)
    else:
        if len(clusterMethods) > 0 and distance_metric not in distance_matrix:
            distance_matrix[distance_metric] = precomputed_matrix(genes_count_unique, distance_metric=distance_metric,
                                                                  tree=tree, otu_ids=otu_ids)

        if len(clusterMethods) > 0 and (
                "k_avg" in clusterMethods or 'bayesian_gaussian_mixture' in clusterMethods) and 'euclidean' not in distance_matrix:
            distance_matrix['euclidean'] = precomputed_matrix(genes_count_unique, distance_metric='euclidean')

        if len(clusterMethods) < 1 and distance_metric not in distance_matrix:
            distance_matrix['euclidean'] = precomputed_matrix(genes_count_unique, distance_metric='euclidean')
            distance_matrix[distance_metric] = precomputed_matrix(genes_count_unique, distance_metric=distance_metric,
                                                                  tree=tree, otu_ids=otu_ids)

    # Извлекаем матрицу признаков один раз (для алгоритмов, работающих с features)
    features_matrix = genes_count_unique.select(
        [c for c in genes_count_unique.columns if c != "Strain"]
    ).to_numpy()

    predictions = []

    for method in clusterMethods:
        if method == 'k_avg':
            # KMeans требует ИСХОДНЫЕ ПРИЗНАКИ
            model = KMeans(n_clusters=int(n_clusters), n_init='auto', random_state=random_state)
            model.fit(features_matrix)
            predictions = model.predict(features_matrix)

        elif method == 'bayesian_gaussian_mixture':
            # GMM тоже требует ИСХОДНЫЕ ПРИЗНАКИ
            adjusted_n_components = adjust_n_components(int(n_clusters), genes_count_unique)
            model = BayesianGaussianMixture(
                n_components=adjusted_n_components, random_state=random_state,
                covariance_type="full", reg_covar=1e-4
            )
            model.fit(features_matrix)
            predictions = model.predict(features_matrix)

        elif method == 'hierarchical_clustering':
            if linkage == "ward":
                # Ward в sklearn работает ТОЛЬКО с исходными признаками
                model = AgglomerativeClustering(n_clusters=int(n_clusters), metric="euclidean", linkage=linkage)
                calc_matrix = features_matrix
            else:
                # Остальные linkage работают с precomputed расстояниями
                model = AgglomerativeClustering(n_clusters=int(n_clusters), metric="precomputed", linkage=linkage)
                calc_matrix = sanitize_distance_matrix(distance_matrix[distance_metric])
            model.fit_predict(calc_matrix)
            predictions = model.labels_

        elif method == 'dbscan':
            model = DBSCAN(eps=float(eps), min_samples=3, metric='precomputed')
            calc_matrix = sanitize_distance_matrix(distance_matrix[distance_metric])
            model.fit(calc_matrix)
            predictions = model.labels_

        elif method == 'spectral_clustering':
            # Spectral с nearest_neighbors/euclidean требует признаки
            model = SpectralClustering(n_clusters=int(n_clusters), affinity=spectral_affinity,
                                       random_state=random_state)
            calc_matrix = features_matrix
            predictions = model.fit_predict(calc_matrix)

        elif method == 'optics':
            kwargs = {"min_samples": 3, "metric": 'precomputed'}
            if optics_max_eps is not None:
                kwargs["max_eps"] = float(optics_max_eps)
            model = OPTICS(**kwargs)
            calc_matrix = sanitize_distance_matrix(distance_matrix[distance_metric])
            predictions = model.fit_predict(calc_matrix)

        elif method == 'affinity_propagation':
            model = AffinityPropagation(damping=damping, preference=None, affinity='precomputed',
                                        random_state=random_state)
            calc_matrix = sanitize_distance_matrix(distance_matrix[distance_metric])
            predictions = model.fit_predict(calc_matrix)

    data.setComputedMatrix(distance_matrix)
    strains = genes_count_unique["Strain"]
    predictions_df = pd.DataFrame(predictions, index=list(strains), columns=["Cluster"])
    data.setCluster(predictions_df)
    return distance_matrix, predictions


def precomputed_matrix(genes_count, distance_metric='euclidean', tree=None, otu_ids=None):
    strains = genes_count["Strain"].to_list()
    numeric_columns = [col for col in genes_count.columns if col != "Strain"]
    genes_count_cut = genes_count.select(numeric_columns).to_numpy()

    if not np.issubdtype(genes_count_cut.dtype, np.number):
        raise ValueError("Counts must be integers or floating-point numbers.")

    return beta_diversity(distance_metric, genes_count_cut, strains)


def statistic_test(data, statMethods, clusterMethods, eps=0.05, distance_metric='euclidean', n_clusters="2",
                   linkage='ward', normalization_method='minmax',
                   tree=None, otu_ids=None, random_state=None):
    genes_count = data.getCount()
    test_result = {}
    errors = []

    if genes_count.is_empty():
        errors.append("данные не выбраны")
        return errors, test_result

    strains = genes_count["Strain"]
    if len(strains) > 1:
        distance_matrix, predictions = clusterization(
            data, clusterMethods=clusterMethods, distance_metric=distance_metric,
            eps=eps, n_clusters=n_clusters, linkage=linkage,
            random_state=random_state, tree=tree, otu_ids=otu_ids
        )
        sample_md = pd.DataFrame(predictions, index=list(strains), columns=["subject"])

        if len(set(predictions)) > 1:
            # Очищаем матрицу для sklearn/skbio
            clean_data = sanitize_distance_matrix(distance_matrix[distance_metric])

            # Восстанавливаем ID штаммов для skbio
            ids = list(distance_matrix[distance_metric].ids) if hasattr(distance_matrix[distance_metric],
                                                                        'ids') else None
            clean_dm = DistanceMatrix(clean_data, ids)

            if 'anosim' in statMethods:
                anosim_result = anosim(clean_dm, sample_md, column='subject', permutations=999)
                if len(anosim_result) > 4 and isinstance(anosim_result[4], (int, float, np.floating)):
                    anosim_result[4] = round(float(anosim_result[4]), 3)
                test_result["ANOSIM"] = list(anosim_result)

            if 'permanova' in statMethods:
                permanova_result = permanova(clean_dm, sample_md, column='subject', permutations=999)
                if len(permanova_result) > 4 and isinstance(permanova_result[4], (int, float, np.floating)):
                    permanova_result[4] = round(float(permanova_result[4]), 3)
                test_result["PERMANOVA"] = list(permanova_result)
        else:
            errors.append(
                "Кластеры в выборке не были выявлены с помощью данного типа кластеризации, попробуйте другой метод.")

        return errors, test_result
    else:
        test_result["Too few strains selected"] = []
        data.setStatResults(test_result)
        return test_result, errors


def match(table_to, table_from):
    for rowIndex, row in table_to.iterrows():
        strain = row['Strain']
        table_from_indexes = table_from.filter(pl.col("Strain") == strain)
        if len(table_from_indexes) > 0:
            table_to.loc[(rowIndex, 'Breakdown Type')] = table_from_indexes[0, 'Breakdown Type'].strip()


def buildScatter(data, components, predictions):
    genes_count = data.getCount()
    components["Strain"] = genes_count["Strain"]
    components['Breakdown Type'] = 'unknown'

    if not data.getBreakdown().is_empty():
        match(components, data.getBreakdown())

    if len(predictions) > 0:
        components['Cluster'] = predictions
        components["Cluster"] = 'Cluster # ' + components["Cluster"].astype(str)
    else:
        components['Cluster'] = 'not predicted'

    def improve_text_position(x):
        positions = ['top center', 'bottom center']
        return [positions[i % len(positions)] for i in range(len(x))]

    fig = px.scatter(components, x='Component 1', y='Component 2', color='Cluster', symbol='Breakdown Type',
                     text='Strain', color_discrete_sequence=px.colors.qualitative.Dark24)
    fig.layout = plotly.graph_objects.Layout(plot_bgcolor='#ffffff', width=900, height=700)
    fig.update_traces(textposition=improve_text_position(components['Component 1']), marker_size=10)
    pltJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return pltJSON


def buildPlots(data, methods, clusterMethods, eps=0.05, perplexity="2", n_clusters='2', linkage='ward',
               distance_metric='euclidean', tree=None, otu_ids=None, random_state=None, normalization_method='minmax',
               spectral_affinity='nearest_neighbors', optics_max_eps=None, damping=0.9):
    plots = {}
    genes_count = data.getCount()

    genes_count = normalize_data(genes_count, method=normalization_method)
    genes_count = genes_count[
        [col for col in genes_count.columns if genes_count[col].null_count() < genes_count.height]]

    if genes_count.height == 0:
        return plots

    strains = genes_count["Strain"]
    features = [col for col in genes_count.columns if col != "Strain"]
    predictions = []

    if len(strains) > 1:
        if len(features) > 1:
            if distance_metric == "euclidean":
                distance_matrix, predictions = clusterization(
                    data, clusterMethods=clusterMethods, eps=eps, n_clusters=n_clusters, linkage=linkage,
                    distance_metric=distance_metric, random_state=random_state, tree=tree, otu_ids=otu_ids,
                    normalization_method=normalization_method, spectral_affinity=spectral_affinity,
                    optics_max_eps=optics_max_eps, damping=damping
                )
            else:
                if "pca" in methods:
                    distance_matrix, predictions = clusterization(
                        data, clusterMethods=clusterMethods, eps=eps, n_clusters=n_clusters, linkage=linkage,
                        distance_metric='euclidean', random_state=random_state, tree=tree, otu_ids=otu_ids,
                        normalization_method=normalization_method, spectral_affinity=spectral_affinity,
                        optics_max_eps=optics_max_eps, damping=damping
                    )
                if 'mds' in methods or "t_sne" in methods:
                    distance_matrix, predictions = clusterization(
                        data, clusterMethods=clusterMethods, eps=eps, n_clusters=n_clusters, linkage=linkage,
                        distance_metric=distance_metric, random_state=random_state, tree=tree, otu_ids=otu_ids,
                        normalization_method=normalization_method, spectral_affinity=spectral_affinity,
                        optics_max_eps=optics_max_eps, damping=damping
                    )

            # PCA работает ТОЛЬКО с исходными признаками
            if 'pca' in methods:
                methodData = PCA(n_components=2, random_state=0)
                pca_features = genes_count.select([c for c in genes_count.columns if c != "Strain"]).to_numpy()
                components = pd.DataFrame(data=methodData.fit_transform(pca_features),
                                          columns=['Component 1', 'Component 2'])
                plots['PCA'] = buildScatter(data, components, predictions)

            # MDS работает с предвычисленными расстояниями
            if 'mds' in methods:
                methodData = MDS(n_components=2, random_state=16, dissimilarity="precomputed")
                x = sanitize_distance_matrix(distance_matrix[distance_metric])
                components = pd.DataFrame(data=methodData.fit_transform(x), columns=['Component 1', 'Component 2'])
                plots['MDS'] = buildScatter(data, components, predictions)

            # t-SNE работает с предвычисленными расстояниями
            if 't_sne' in methods:
                init = "pca" if distance_metric == "euclidean" else "random"
                methodData = TSNE(
                    random_state=0, perplexity=float(perplexity),
                    metric="precomputed" if distance_metric != "euclidean" else "euclidean", init=init
                )
                x = sanitize_distance_matrix(distance_matrix[distance_metric])
                components = pd.DataFrame(data=methodData.fit_transform(x), columns=['Component 1', 'Component 2'])
                plots['t-SNE'] = buildScatter(data, components, predictions)
        else:
            x, y = [], []
            for value in genes_count.select(features).to_numpy():
                x.append(value[0])
                y.append(0)
            components = pd.DataFrame(data={'Component 1': x, 'Component 2': y})
            plots['No method'] = buildScatter(data, components, strains)

    return plots