import pandas as pd
import os
from datetime import datetime
import lxml.etree
import lxml.builder
import gzip

def create_gretl_export(model, filename=None):
    # Create filename with timestamp if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simulation_report_{timestamp}.gdt"
    
    # Ensure the output directory exists
    os.makedirs('output', exist_ok=True)
    filepath = os.path.join('output', filename)

    # Create history DataFrame
    df_history = pd.DataFrame(model.history)
    
    # Create current state DataFrames
    companies_data = [{
        'ID': i+1,
        'Capitale (K)': imp.K,
        'Patrimonio Netto (A)': imp.A,
        'Debito (L)': imp.L,
        'Profitto (π)': imp.pi,
        'Tasso Interesse (r)': imp.r
    } for i, imp in enumerate(model.impresa)]
    
    df_companies = pd.DataFrame(companies_data)
    
    # Summary statistics
    summary_data = {
        'Metriche': [
            'Numero Imprese',
            'Produzione Totale',
            'Capitale Totale',
            'Debito Totale',
            'Patrimonio Netto Banca',
            'Profitto Banca',
            'Credito Totale Disponibile'  # Aggiunta questa riga
        ],
        'Valori': [
            len(model.impresa),
            model.y,
            sum(f.K for f in model.impresa),
            sum(f.L for f in model.impresa),
            model.PatrimonioNettoBanca,
            model.profittoBanca,
            10 * model.PatrimonioNettoBanca  # Aggiunta questa riga
        ]
    }
    #df_summary = pd.DataFrame(summary_data)

    E = lxml.builder.ElementMaker()
    gretl_data = E.gretldata
    DESCRIPTION = E.description
    VARIABLES = E.variables
    VARIABLE = E.variable
    OBSERVATIONS = E.observations
    OBS = E.obs
    variables = VARIABLES(count='{}'.format( sum((1 for _ in df_history ))))
    observations = OBSERVATIONS(count='{}'.format(len(df_history)), labels='false')
    for variable_name in df_history:
        variables.append(VARIABLE(name='{}'.format(variable_name.replace(" ","_"))))
    for i in range(len(df_history)):
        string_obs = ''
        for variable_name in df_history:
            string_obs += '{}  '.format(df_history[variable_name][i])
        observations.append(OBS(string_obs))
    gdt_result = gretl_data(DESCRIPTION('gretl'), variables, observations, version='1.4', name='simulatore',
                            frequency='special:1', startobs='1', endobs='{}'.format(len(df_history)),
                            type='time-series')
    with gzip.open(filename, 'w') as output_file:
        output_file.write(b'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE gretldata SYSTEM "gretldata.dtd">\n')
        output_file.write(lxml.etree.tostring(gdt_result, pretty_print=True, encoding=str).encode('ascii'))


    
    print(f"salvato in: {filepath}")
    return filepath