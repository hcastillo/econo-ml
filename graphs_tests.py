from progress.bar import Bar

import interbank
import interbank_lenderchange
import networkx as nx
import os

NUMBER_ELEMENTS = 10
NUMBER_OF_BANKS = 200
OUTPUT = "graphs_tests/"
model = interbank.Model()
progress_bar = Bar("Getting graphs", max=NUMBER_ELEMENTS*2)
progress_bar.update()
for i in range(NUMBER_ELEMENTS):
    for type_of_graph in ('barabasi', 'erdos_renyi','smallworld'):
        output_folder = OUTPUT + f'{type_of_graph}/'

        if not os.path.exists(f'{output_folder}'):
            os.mkdir(f'{output_folder}')
        if os.path.exists(f'{output_folder}/banks_{i}.json'):
            continue

        try:
            match type_of_graph:
                case 'barabasi':
                    model = interbank.Model()
                    model.config.lender_change = interbank_lenderchange.Preferential()
                    model.config.lender_change.set_parameter("m", (1 + i * 10))
                    model.configure(T=5, N=NUMBER_OF_BANKS)
                    model.initialize(export_datafile=None, generate_plots=False)
                    result = model.config.lender_change.initialize_bank_relationships(model)
                    origin = model.config.lender_change.banks_graph_full
                    origin1 = nx.barabasi_albert_graph(n=NUMBER_OF_BANKS, m=(1 + i * 10))
                    description = f"barabasi m={1 + i * 10}"
                case 'erdos_renyi':
                    model = interbank.Model()
                    p = 0.003 + i/1000
                    model.config.lender_change = interbank_lenderchange.ShockedMarket()
                    model.config.lender_change.set_parameter("p", p)
                    model.configure(T=5, N=NUMBER_OF_BANKS)
                    model.initialize(export_datafile=None, generate_plots=False)
                    result = model.config.lender_change.initialize_bank_relationships(model)
                    origin1 = None  # model.config.lender_change.banks_graph_full
                    origin = nx.erdos_renyi_graph(n=NUMBER_OF_BANKS, p=(p))
                    description = f"erdos renyi p={p}"
                case _:
                    origin = nx.watts_strogatz_graph(NUMBER_OF_BANKS, 10, p=(i / 10))
                    result = interbank_lenderchange.SmallWorld.create_directed_graph_from_watts_strogatz(origin)
                    description = f"watts strogatz p={i / 10}"
                    origin1 = None
        except:
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