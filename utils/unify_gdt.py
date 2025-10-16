#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Input: directories where we have a file called results.gdt and same variables
Output: gdt file with the same variables but in a single file, for instance:

    dir1\results.gdt - bankruptcies,
    dir2\results.gdt - bankruptcies,equity,

    output.gdt - bankruptcies_dir1, bankruptcies_dir2, equity_dir2

@author: hector@bith.net
@date:   10/2025
"""

import lxml.etree
import lxml.builder
import os
import argparse
import gzip
import sys
from interbank import Statistics

WORKING_DIR = '/experiments'
OUTPUT_FILE = 'output.gdt'


class GretlUnifier:
    # renames can contain key values as { 'dir1':'d' } so resultant variable will be
    # bankruptcies_d instead of bankruptcies_dir1:
    RENAMES_DIRS = {'-not_exists-': '_n'}

    REPLACES_COLUMNS = {'psi=':''}

    @staticmethod
    def transform_line_from_string(line_with_values):
        items = []
        for i in line_with_values.replace('  ', ' ').strip().split(' '):
            try:
                items.append(int(i))
            except ValueError:
                items.append(float(i))
        return items

    def save_gdt(self, data, filename, header_text):
        element = lxml.builder.ElementMaker()
        gretl_data = element.gretldata
        xml_description = element.description
        xml_variables = element.variables
        variable = element.variable
        xml_observations = element.observations
        observation = element.obs
        variables = xml_variables(count='{}'.format(len(data.columns)))
        for i, variable_name in enumerate(data.columns):
            if variable_name == 'leverage':
                variable_name += '_'
            if i == 0:
                variables.append(variable(name='{}'.format(variable_name), label='{}'.format(header_text)))
            else:
                variables.append(variable(name='{}'.format(variable_name)))
        xml_observations = xml_observations(count='{}'.format(len(data)), labels='false')
        for i in range(len(data)):
            string_obs = ''
            for variable in data.columns:
                string_obs += '{}  '.format(data[variable][i])
            xml_observations.append(observation(string_obs))
        gdt_result = gretl_data(xml_description(header_text), variables,
                                xml_observations, version='1.4', name='interbank',
                                frequency='special:1', startobs='1', endobs='{}'.format(len(data)),
                                type='time-series')
        with gzip.open(filename, 'w') as output_file:
            output_file.write(b'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE gretldata SYSTEM "gretldata.dtd">\n')
            output_file.write(lxml.etree.tostring(gdt_result, pretty_print=True, encoding=str).encode('ascii'))
        print(f"{filename} created")

    def __determine_new_name(self, col, experiments, experiment, results):
        new_name = col + '_' + experiments[experiment].replace('.','')
        for item in self.REPLACES_COLUMNS:
            new_name = new_name.replace(item, self.REPLACES_COLUMNS[item])
        num = 0
        while len(new_name) > 30 or (results is not None and new_name in results.columns):
            new_name = new_name[:28] + (str(num) if num > 0 else '')
            num += 1
        if num > 0:
            print(f"column {col}_{experiments[experiment]} named to {new_name} due to collision or length")
        return new_name

    def error(self, message):
        print(message, file=sys.stderr)
        sys.exit(0)

    def unify(self, working_dir: str, input_experiments: list, output_file: str):
        if os.path.isfile(output_file):
            self.error(f"output file {output_file} already exists")
        experiments = {}
        for args_item in input_experiments:
            for item in args_item.strip().split(" "):
                if item in self.RENAMES_DIRS:
                    experiments[item] = self.RENAMES_DIRS[item][1:] \
                        if self.RENAMES_DIRS[item].startswith('_') else self.RENAMES_DIRS[item]
                else:
                    if ':' in item:
                        item_split = item.split(':')
                        experiments[item_split[0]] = item_split[1][1:] \
                            if item_split[1].startswith('_') else item_split[1]
                    else:
                        experiments[item] = item

        header = ''
        results = None
        for experiment in experiments:
            file = None
            if os.path.isfile(experiment):
                file = experiment
            elif os.path.isfile(experiment + ".gdt"):
                file = experiment + '.gdt'
            else:
                file = working_dir + '/' + experiment + '/results.gdt'
                if not os.path.isfile(file):
                    self.error(file + " not found (%s)" % experiment)
            if file:
                data , _ = Statistics.read_gdt(file)
                if experiments[experiment][1:] == experiment:
                    header += f"{experiment} "
                else:
                    header += f"{experiment}:" + experiments[experiment] + " "
                for col in data.columns:
                    data.rename(columns={col: self.__determine_new_name(col, experiments, experiment, results)},
                                inplace=True)
                if results is None:
                    results = data
                else:
                    if len(data) != len(results):
                        self.error(f"{file} has {len(data)} rows and previous {len(results)}")
                    results = results.join(data, how='outer')
        self.save_gdt(results, working_dir+"/"+output_file, header)


def run_interactive():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action='append', help='<Required> List of executions to unify. '
                                                         'A value dir1:n will be saved as _n instead of _dir',
                        required=True)
    parser.add_argument("--output", default=OUTPUT_FILE, help=f"Output GDT file (default {OUTPUT_FILE})")
    parser.add_argument("--working_dir", default=WORKING_DIR,
                        help=f"Directory where executions are located (default {WORKING_DIR})")
    args = parser.parse_args()
    gretl = GretlUnifier()
    gretl.unify(working_dir=args.working_dir, input_experiments=args.input, output_file=args.output)


if __name__ == "__main__":
    run_interactive()
