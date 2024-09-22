
import interbank_lenderchange
import networkx as nx

erdos = interbank_lenderchange.load_graph_json('erdos.json')
bara = interbank_lenderchange.load_graph_json('bara.json')
small = interbank_lenderchange.load_graph_json('small.json')
restricted = interbank_lenderchange.load_graph_json('restricted.json')
