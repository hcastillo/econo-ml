from progress.bar import Bar

import interbank
import interbank_lenderchange
import networkx as nx
import os

NUMBER_ELEMENTS = 10
NUMBER_OF_BANKS = 50
TYPES_OF_GRAPHS = ('barabasi', 'erdos_renyi', 'smallworld')
OUTPUT = "../graphs_tests/"
model = interbank.Model()
progress_bar = Bar("Getting graphs", max=NUMBER_ELEMENTS*len(TYPES_OF_GRAPHS))
progress_bar.update()
for i in range(NUMBER_ELEMENTS):
    for type_of_graph in TYPES_OF_GRAPHS:
        output_folder = OUTPUT + f'{type_of_graph}'
        if not os.path.exists(f'{output_folder}'):
            os.mkdir(f'{output_folder}')
        if os.path.exists(f'{output_folder}/banks_{i}.json'):
            continue
        model = interbank.Model()
        origin = None
        description = None
        origin1 = None
        result = None
        try:
            match type_of_graph:
                case 'barabasi':
                    model.config.lender_change = interbank_lenderchange.Preferential()
                    m = int(NUMBER_OF_BANKS/NUMBER_ELEMENTS*i)
                    if m==0:
                        m=1
                    model.config.lender_change.set_parameter("m", m)
                    model.configure(T=5, N=NUMBER_OF_BANKS)
                    model.initialize(export_datafile=None, generate_plots=False)
                    result = model.config.lender_change.initialize_bank_relationships(model)
                    origin = model.config.lender_change.banks_graph_full
                    origin1 = nx.barabasi_albert_graph(n=NUMBER_OF_BANKS, m=m)
                    description = f"barabasi m={m}"
                case _:
                    p = i / NUMBER_ELEMENTS
                    if p == 0:
                        p = 0.003
                    elif p == 1:
                        p = 0.999
                    if type_of_graph == 'erdos_renyi':
                        model.config.lender_change = interbank_lenderchange.ShockedMarket()
                        origin1 = nx.erdos_renyi_graph(n=NUMBER_OF_BANKS, p=p)
                        description = f"erdos renyi p={p}"
                    else:
                        model.config.lender_change = interbank_lenderchange.SmallWorld()
                        origin1 = nx.watts_strogatz_graph(n=NUMBER_OF_BANKS, k=1, p=p)
                        description = f"watts strogatz p={p}"
                    model.config.lender_change.set_parameter("p", p)
                    model.configure(T=5, N=NUMBER_OF_BANKS)
                    model.initialize(export_datafile=None, generate_plots=False)
                    result = model.config.lender_change.initialize_bank_relationships(model)
                    origin = model.config.lender_change.banks_graph
        except SyntaxError:
            pass
        else:
            interbank_lenderchange.save_graph_png(origin, description,
                                                  f'{output_folder}/{i}.png', add_info=True)
            interbank_lenderchange.save_graph_json(origin, f'{output_folder}/{i}.json')
            interbank_lenderchange.save_graph_png(result, description,
                                                  f'{output_folder}/banks_{i}.png', add_info=True)
            interbank_lenderchange.save_graph_json(result, f'{output_folder}/banks_{i}.json')
            if origin1:
                interbank_lenderchange.save_graph_png(origin1, description, f'{output_folder}/aux_{i}.png',
                                                      add_info=True)
                interbank_lenderchange.save_graph_json(origin1, f'{output_folder}/aux_{i}.json')
        progress_bar.next()
progress_bar.finish()

