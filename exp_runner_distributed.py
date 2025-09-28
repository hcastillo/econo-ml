#!/usr/bin/env python
# coding: utf-8
"""
Executor base class for the interbank model
@author: hector@bith.net
"""
import exp_runner
import ray
import pandas as pd
import time
from progress.bar import Bar
import sys


@ray.remote
def actor_combination_execution(model_configuration, model_parameters,
                                clear_previous_results, seeds_for_random,
                                position_inside_seeds_for_random,
                                filename_for_iteration,
                                experiment:exp_runner.ExperimentRun):
    experiment.verify_directories()
    result_iteration_to_check = pd.DataFrame()
    graphs_iteration = []
    # first round to load all the self.MC and estimate mean and standard deviation of the series
    # inside result_iteration:
    for i in range(experiment.MC):
        result_mc = experiment.load_or_execute_model( model_configuration, model_parameters,
                                      filename_for_iteration, i,
                                      clear_previous_results,
                                      seeds_for_random[i + position_inside_seeds_for_random])
        graphs_iteration.append(f"{experiment.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}")
        result_iteration_to_check = pd.concat([result_iteration_to_check,result_mc])

    # second round to verify if one of the models should be replaced because it presents abnormal
    # values comparing to the other (self.MC-1) stored in result_iteration_to_check:
    result_iteration = pd.DataFrame()
    position_inside_seeds_for_random -= experiment.MC
    for i in range(experiment.MC):
        result_mc = experiment.load_model_and_rerun_till_ok(
                                      model_configuration, model_parameters, filename_for_iteration,
                                      i, clear_previous_results, seeds_for_random,
                                      position_inside_seeds_for_random, result_iteration_to_check)
        result_iteration = pd.concat([result_iteration, result_mc])

    results = {}
    for k in result_iteration.keys():
        if k.strip() == "t":
            continue
        mean_estimated = result_iteration[k].mean()
        std_estimated = result_iteration[k].std()
        results[k] = [mean_estimated, std_estimated]
    return results


class ExperimentRun(exp_runner.ExperimentRun):
    is_distributed = True

class Runner(exp_runner.Runner):

    RAY_DEFAULT_PORT = 10001
    ray_connected = None

    def set_ray_cluster(self, head_ip: str):
        if not head_ip.lower().startswith("ray://"):
            head_ip = "ray://" + head_ip
        if not ':' in head_ip.replace('ray://', '').lower():
            head_ip = head_ip + f':{self.RAY_DEFAULT_PORT}'
        try:
            if not self.ray_connected:
                self.ray_connected = ray.init(head_ip)
        except:
            print(f"Unreachable or bad head ip. Try: 'ip:{self.RAY_DEFAULT_PORT}'")
            sys.exit(-1)


    def run_execution_distributed(self, experiment:exp_runner.ExperimentRun, clear_previous_results:bool=False,
                                  reverse_execution:bool=False):
        experiment.log_replaced_data = ""
        initial_time = time.perf_counter()
        if clear_previous_results:
            results_to_plot = {}
            results_x_axis = []
        else:
            results_to_plot, results_x_axis = experiment.load(f"{experiment.OUTPUT_DIRECTORY}/")
        if not results_to_plot:
            seeds_for_random = experiment.generate_random_seeds_for_this_execution()
            progress_bar = Bar(
                "Executing models", max=experiment.get_num_models()
            )
            progress_bar.update()
            position_inside_seeds_for_random = 0
            results_to_plot = {}
            results_x_axis = []
            futures = []

            array_of_configs = self.get_models(self.config)
            array_of_parameters = self.get_models(self.parameters)
            if reverse_execution:
                array_of_configs = reversed(list(array_of_configs))
                array_of_parameters = reversed(list(array_of_parameters))

            for config_i in array_of_configs:
                for parameter_j in array_of_parameters:
                    futures.append( actor_combination_execution.remote(config_i, parameter_j,
                                                                       clear_previous_results, seeds_for_random,
                                                                       position_inside_seeds_for_random,
                                                                       experiment.get_filename_for_iteration(
                                                                           config_i, parameter_j),
                                                                       experiment))
                    results_x_axis.append(experiment.get_title_for(config_i, parameter_j))

            for results in ray.get(futures):
                for k in results.keys():
                    if k in results_to_plot:
                        results_to_plot[k].append(results[k])
                    else:
                        results_to_plot[k] = [results[k]]
                progress_bar.next()
            progress_bar.finish()
            print(f"Saving results in {experiment.OUTPUT_DIRECTORY}...")
            experiment.save_csv(results_to_plot, results_x_axis, f"{experiment.OUTPUT_DIRECTORY}/")
            experiment.save_gdt(results_to_plot, results_x_axis, f"{experiment.OUTPUT_DIRECTORY}/")
        else:
            print(f"Loaded data from previous work from {experiment.OUTPUT_DIRECTORY}")
        results_comparing = experiment.load_comparing(results_to_plot, results_x_axis)
        if experiment.log_replaced_data:
            print(experiment.log_replaced_data)
        print("Plotting...")
        experiment.plot(results_to_plot, results_x_axis, experiment.get_title_for(experiment.config, experiment.parameters),
                  f"{experiment.OUTPUT_DIRECTORY}/", results_comparing)
        experiment.results_to_plot = results_to_plot
        final_time = time.perf_counter()
        print('execution_time: %2.5f secs' % (final_time - initial_time))
        return results_to_plot, results_x_axis

    def do(self):
        self.parser.add_argument(
            "--ray",
            default=None,
            type=str,
            help="Distribute the load between nodes of a Ray cluster (ray://ip:10001)",
        )

        args = self.parser.parse_args()
        experiment = self.experiment_runner()
        if args.clear_results:
            experiment.clear_results()
        experiment.error_bar = args.errorbar
        if args.ray:
            self.set_ray_cluster(args.ray)
        if args.listnames:
            experiment.listnames()
        elif args.do:
            if self.ray_connected:
                self.run_execution_distributed(experiment=experiment,
                                               clear_previous_results=args.clear, reverse_execution=args.reverse)
            else:
                experiment.do(clear_previous_results=args.clear, reverse_execution=args.reverse)
            return experiment
        else:
            self.parser.print_help()


