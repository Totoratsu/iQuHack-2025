def interpret_solution(graph, bitstring):
    """
    Visualize the given ``bitstring`` as a partition of the given ``graph``.
    """
    pos = nx.spring_layout(graph, seed=42)
    set_0 = [i for i, b in enumerate(bitstring) if b == "0"]
    set_1 = [i for i, b in enumerate(bitstring) if b == "1"]

    plt.figure(figsize=(4, 4))
    nx.draw_networkx_nodes(
        graph, pos=pos, nodelist=set_0, node_color="blue", node_size=700
    )
    nx.draw_networkx_nodes(
        graph, pos=pos, nodelist=set_1, node_color="red", node_size=700
    )

    cut_edges = []
    non_cut_edges = []
    for u, v in graph.edges:
        if bitstring[u] != bitstring[v]:
            cut_edges.append((u, v))
        else:
            non_cut_edges.append((u, v))

    nx.draw_networkx_edges(
        graph, pos=pos, edgelist=non_cut_edges, edge_color="gray", width=2
    )
    nx.draw_networkx_edges(
        graph, pos=pos, edgelist=cut_edges, edge_color="green", width=2, style="dashed"
    )

    nx.draw_networkx_labels(graph, pos=pos, font_color="white", font_weight="bold")
    plt.axis("off")
    plt.show()
