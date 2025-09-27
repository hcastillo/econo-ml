#!/usr/bin/env python
# coding: utf-8
"""
Executor base class for the interbank model
@author: hector@bith.net
"""
import exp_runner
import ray
import concurrent.futures
import os
import pandas as pd
import interbank
import time
from progress.bar import Bar


def run_model(filename, execution_config, execution_parameters, seed_random):
    model = interbank.Model()
    model.export_datafile = filename
    model.config.lender_change = self.set_lender_change(execution_parameters)
    model.configure(T=self.T, N=self.N,
                    allow_replacement_of_bankrupted=self.ALLOW_REPLACEMENT_OF_BANKRUPTED, **execution_config)
    model.configure(**self.EXTRA_MODEL_CONFIGURATION)
    model.initialize(seed=seed_random, save_graphs_instants=None,
                     export_datafile=filename,
                     generate_plots=False)
    model.simulate_full(interactive=False)
    return model.finish()

def load_or_execute_model(model_configuration, model_parameters, filename_for_iteration,
                          output_directory,
                          i, clear_previous_results, seed_for_this_model):
    if (os.path.isfile(f"{output_directory}/{filename_for_iteration}_{i}.csv")
            and not clear_previous_results):
        result_mc = pd.read_csv(
            f"{output_directory}/{filename_for_iteration}_{i}.csv", header=2)
    elif (os.path.isfile(f"{output_directory}/{filename_for_iteration}_{i}.gdt")
          and not clear_previous_results):
        result_mc = interbank.Statistics.read_gdt(
            f"{output_directory}/{filename_for_iteration}_{i}.gdt")
    else:
        result_mc = run_model(
            f"{output_directory}/{filename_for_iteration}_{i}",
            model_configuration, model_parameters, seed_for_this_model)
    return result_mc


def load_model_and_rerun_till_ok(model_configuration, model_parameters, filename_for_iteration,
                                     output_directory, i, clear_previous_results, seeds_for_random,
                                     position_inside_seeds_for_random, result_iteration_to_check):
    """
            Internal function in which we have result_iteration_to_check with the averages of the MC iterations, and
            we check individually if any of those individual executions is an outlier, we replace it using a different
            seed and incorporate to the results:
            """
    result_mc = load_or_execute_model(model_configuration, model_parameters,
                                      filename_for_iteration, output_directory, i,
                                      clear_previous_results,
                                      seeds_for_random[i + position_inside_seeds_for_random])
    offset = 1
    while not data_seems_ok(filename_for_iteration, i, result_mc, result_iteration_to_check) \
            and offset <= MAX_EXECUTIONS_OF_MODELS_OUTLIERS:
        discard_execution_of_iteration(filename_for_iteration, i)
        result_mc = load_or_execute_model(model_configuration, model_parameters,
                                          filename_for_iteration, output_directory, i,
                                          clear_previous_results,
                                          (seeds_for_random[i + position_inside_seeds_for_random]+ offset))
        offset += 1
    return result_mc


@ray.remote
def actor_combination_execution(model_configuration, model_parameters,
                                clear_previous_results, seeds_for_random,
                                position_inside_seeds_for_random,
                                filename_for_iteration, output_directory, mc):
    result_iteration_to_check = pd.DataFrame()
    graphs_iteration = []
    # first round to load all the self.MC and estimate mean and standard deviation of the series
    # inside result_iteration:
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results_mc = {executor.submit(load_or_execute_model,
                                      model_configuration, model_parameters,
                                      filename_for_iteration, output_directory, i,
                                      clear_previous_results,
                                      seeds_for_random[i + position_inside_seeds_for_random]):
                          i for i in range(mc)}
        for future in concurrent.futures.as_completed(results_mc):
            i = results_mc[future]
            graphs_iteration.append(f"{output_directory}/{filename_for_iteration}_{i}")
            result_iteration_to_check = pd.concat([result_iteration_to_check, future.result()])

    # second round to verify if one of the models should be replaced because it presents abnormal
    # values comparing to the other (self.MC-1) stored in result_iteration_to_check:
    result_iteration = pd.DataFrame()
    position_inside_seeds_for_random -= mc
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results_mc = {executor.submit(load_model_and_rerun_till_ok,
                                      model_configuration, model_parameters, filename_for_iteration,
                                      output_directory, i, clear_previous_results, seeds_for_random,
                                      position_inside_seeds_for_random, result_iteration_to_check):
                          i for i in range(mc)}
        for future in concurrent.futures.as_completed(results_mc):
            result_iteration = pd.concat([result_iteration, future.result()])

    results = {}
    for k in result_iteration.keys():
        if k.strip() == "t":
            continue
        mean_estimated = result_iteration[k].mean()
        std_estimated = result_iteration[k].std()
        results[k] = [mean_estimated, std_estimated]
    return results


class ExperimentRunDistributed(exp_runner.ExperimentRun):

    RAY_DEFAULT_PORT = 10001
    ray_connected = None

    def set_ray_cluster(self, head_ip: str):
        if not head_ip.lower().startswith("ray://"):
            head_ip = "ray://" + head_ip
        if not ':' in head_ip.replace('ray://', '').lower():
            head_ip = head_ip + f':{self.RAY_DEFAULT_PORT}'
        #try:
        if not self.ray_connected:
                self.ray_connected = ray.init(head_ip, runtime_env={"working_dir":"/home/hector/econo-ml"})
        #except:
        #    print(f"Unreachable or bad head ip. Try: 'ip:{self.RAY_DEFAULT_PORT}'")
        #    sys.exit(-1)


    def generate_distributed_execution(self, model_configurations, model_parameters,
                                       clear_previous_results, seeds_for_random,
                                       position_inside_seeds_for_random,
                                       progress_bar):
        results_to_plot = {}
        results_x_axis = []
        futures = [actor_combination_execution.remote(model_i, parameter_j,
                                clear_previous_results, seeds_for_random,
                                position_inside_seeds_for_random,
                                self.get_filename_for_iteration(model_i, parameter_j),
                                self.OUTPUT_DIRECTORY, self.MC)
                   for model_i in model_configurations for parameter_j in model_parameters]

        for results in ray.get(futures):
            for k in results.keys():
                if k in results_to_plot:
                    results_to_plot[k].append(results[k])
                else:
                    results_to_plot[k] = [results[k]]
            progress_bar.next()
        return results_to_plot, results_x_axis


    def do(self, clear_previous_results=False):
        if not self.ray_connected:
            print("no ray cluster configured: executing locally")
            return super().do(clear_previous_results)
        else:
            self.log_replaced_data = ""
            initial_time = time.perf_counter()
            if clear_previous_results:
                results_to_plot = {}
                results_x_axis = []
            else:
                results_to_plot, results_x_axis = self.load(f"{self.OUTPUT_DIRECTORY}/")
            if not results_to_plot:
                self.__verify_directories__()
                seeds_for_random = self.generate_random_seeds_for_this_execution()
                progress_bar = Bar(
                    "Executing models", max=self.get_num_models()
                )
                progress_bar.update()
                position_inside_seeds_for_random = 0
                results_to_plot, results_x_axis = self.generate_distributed_execution(
                                       self.get_models(self.config), self.get_models(self.parameters),
                                       clear_previous_results, seeds_for_random,
                                       position_inside_seeds_for_random, progress_bar)
                progress_bar.finish()
                print(f"Saving results in {self.OUTPUT_DIRECTORY}...")
                self.save_csv(results_to_plot, results_x_axis, f"{self.OUTPUT_DIRECTORY}/")
                self.save_gdt(results_to_plot, results_x_axis, f"{self.OUTPUT_DIRECTORY}/")
            else:
                print(f"Loaded data from previous work from {self.OUTPUT_DIRECTORY}")
            results_comparing = self.load_comparing(results_to_plot, results_x_axis)
            if self.log_replaced_data:
                print(self.log_replaced_data)
            print("Plotting...")
            self.plot(results_to_plot, results_x_axis, self.__get_title_for(self.config, self.parameters),
                      f"{self.OUTPUT_DIRECTORY}/", results_comparing)
            self.results_to_plot = results_to_plot
            final_time = time.perf_counter()
            print('execution_time: %2.5f secs' % (final_time - initial_time))
            return results_to_plot, results_x_axis


class Runner(exp_runner.Runner):
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
            experiment.set_ray_cluster(args.ray)
        if args.listnames:
            experiment.listnames()
        elif args.do:
            experiment.do(args.clear)
            return experiment
        else:
            self.parser.print_help()


