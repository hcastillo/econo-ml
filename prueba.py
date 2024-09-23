
import interbank_lenderchange
import networkx as nx
import matplotlib.pyplot as plt

# erdos = interbank_lenderchange.load_graph_json('erdos.json')
# bara = interbank_lenderchange.load_graph_json('bara.json')
# small = interbank_lenderchange.load_graph_json('small.json')
# result = interbank_lenderchange.SmallWorld.prueba6(small)
# node_positions = nx.spring_layout(small)
# plt.clf()
# nx.draw(small, pos=node_positions, with_labels=True)
# plt.savefig(f'small.png')
# interbank_lenderchange.save_graph_json(small, f'small_d.json')
# plt.clf()
# nx.draw(result, pos=node_positions, with_labels=True, arrowstyle='->')
# # nx.draw_networkx_edge_labels(dd, pos=node_positions)
# plt.savefig(f'small_d.png')


for i in range(30):
    d = nx.watts_strogatz_graph(50,2, p=i/100)
    d.remove_node(40)
    dd = interbank_lenderchange.SmallWorld.prueba6(d)
    interbank_lenderchange.save_graph_json(d,f'small/{i}.json')
    plt.clf()
    node_positions = nx.spring_layout(d)
    nx.draw(d, pos=node_positions, with_labels=True)
    plt.savefig(f'small/{i}.png')
    interbank_lenderchange.save_graph_json(dd,f'small/{i}_d.json')
    plt.clf()
    nx.draw(dd, pos=node_positions, with_labels=True, arrowstyle='->')
    #nx.draw_networkx_edge_labels(dd, pos=node_positions)
    plt.savefig(f'small/{i}_d.png')

# a = nx.Graph()
# a.add_edge(4,3)
# a.add_edge(3,2)
# a.add_edge(2,0)
# a.add_edge(0,1)
# a.add_edge(1,2)
# aa = interbank_lenderchange.SmallWorld.prueba(a)
# nx.draw(a, with_labels=True)
# plt.show()
