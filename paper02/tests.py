def test_cluster_evaluate():
    from src.datasets import alldts
    from src.metrics import cluster_evaluate

    selection = {"cred_aus", "cred_ger", "breast_coimbra", "sonar", "heart"}
    datasets = [(k, v) for k, v in alldts().items() if k in selection]
    aux = list()
    for dataset_name, (data, target) in datasets:
        aux.append(dict({"dataset": dataset_name}, **cluster_evaluate(data.values, target.values)))


def test_gabriel_graph():
    from src.graph import GabrielGraph
    from src.datasets import get_linear

    data, target = get_linear(n_obs=100)
    gg = GabrielGraph(X=data)
    gg.adjacency()

    gg.plot()


def test_graph_quality():
    from src.datasets import get_blobs
    from src.metrics import GGMetrics

    data, target = get_blobs(n_obs=100)

    ggm = GGMetrics(data, target)
    ggm.gg_border_perc()
    ggm.gg_class_quality()
