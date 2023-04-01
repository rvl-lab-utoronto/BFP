import numpy as np
import pandas as pd
import tqdm
import wandb

def prob_runs(
        project_name="csc2541-cl/csc2541-cl",
        filters=None,
        ):
    '''
    Print and return a list the experiment runs using the given filters.
    Used for verifying the filters before actually pulling all the data. 
    '''
    api = wandb.Api(timeout=60)
    runs = api.runs(project_name, filters=filters)

    print("Found {} runs.".format(len(runs)), end='')
    print("Names of the runs:")
    for run in runs:
        print(run.id, run.name)

    return runs

def wandb_get_history(
        keys, 
        project_name="csc2541-cl/csc2541-cl",
        filters=None,
        ):
    '''
    Pull the experiment loggings from wandb.
    The columns are specified in keys, which must match the full name in wandb. 
    some basic information about the run will also be returned. 
    '''
    api = wandb.Api(timeout=60)
    runs = api.runs(project_name, filters=filters)

    print("Found {} runs.".format(len(runs)))
    # print("Names of the runs:")
    # for run in runs:
    #     print(run.id, run.name)
    
    keys = keys + ['_step']
    results = []
    for run in tqdm.tqdm(runs):
        id = run.id
        name = run.name
        model = run.config['model']
        dataset = run.config['dataset']
        try:
            buffer_size = run.config['buffer_size']
        except KeyError:
            buffer_size = None

        for row in run.scan_history(keys=keys):
            datapoint = {
                "id": id,
                "name": name,
                "model": model,
                "dataset": dataset,
                "buffer_size": buffer_size,
            }
            datapoint.update(row)
            results.append(datapoint)

    df = pd.DataFrame.from_dict(results)

    return df


def wandb_get_meta(
        project_name="csc2541-cl/csc2541-cl",
        filters=None,
        ):
    '''
    Pull the meta/aggregated information from wandb project, includiing:
    * Summary of each logged metric
    * Configuration of the run
    * Name of the run
    * id of the run
    '''
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs(project_name, filters=filters)
    summary_list = [] 
    config_list = [] 
    name_list = []
    id_list = []
    for run in runs: 
        # run.summary are the output key/values like accuracy.
        # We call ._json_dict to omit large files 
        summary_list.append(run.summary._json_dict) 

        # run.config is the input metrics.
        # We remove special values that start with _.
        config = {k:v for k,v in run.config.items() if not k.startswith('_')}
        config_list.append(config) 

        # run.name is the name of the run.
        name_list.append(run.name)

        id_list.append(run.id)

    summary_df = pd.DataFrame.from_records(summary_list) 
    config_df = pd.DataFrame.from_records(config_list) 
    name_df = pd.DataFrame({'name': name_list}) 
    id_df = pd.DataFrame({'id': id_list}) 
    all_df = pd.concat([id_df, name_df, config_df,summary_df], axis=1)

    return all_df


def main_table(df, config, metrics, latex_meanstd=True):
    df = df.groupby(config, as_index=False).agg({
        k: ['mean', 'std'] for k in metrics
    })

    for k in metrics:
        if latex_meanstd:
            df[k+'_'] = "\\meanstd{" + df[k]['mean'].apply(lambda x: f"{x:.2f}") + "}{" + df[k]['std'].apply(lambda x: f"{x:.2f}") + "}"
        else:
            df[k+"_"] = df[k]['mean'].apply(lambda x: f"{x:.2f}") + "±" + df[k]['std'].apply(lambda x: f"{x:.2f}")
        df.drop(columns=[k], inplace=True)
        df[k] = df[k+"_"]
        df.drop(columns=[k+"_"], inplace=True)

    keys = config + metrics
    df = df[keys]
    df.loc[(df['base_method']=='ER') & (df['proj_type']=='none'), 'model'] = 'ER'
    df.loc[(df['base_method']=='ER') & (df['proj_type']=='0'), 'model'] = 'ER w/ FD'
    df.loc[(df['base_method']=='ER') & (df['proj_type']=='1'), 'model'] = 'ER w/ BFP'
    df.loc[(df['base_method']=='ER') & (df['proj_type']=='2'), 'model'] = 'ER w/ BFP-2'
    df.loc[(df['base_method']=='DER++') & (df['proj_type']=='none'), 'model'] = 'DER++'
    df.loc[(df['base_method']=='DER++') & (df['proj_type']=='0'), 'model'] = 'DER++ w/ FD'
    df.loc[(df['base_method']=='DER++') & (df['proj_type']=='1'), 'model'] = 'DER++ w/ BFP'
    df.loc[(df['base_method']=='DER++') & (df['proj_type']=='2'), 'model'] = 'DER++ w/ BFP-2'
    df = df[df['buffer_size'] != 1000]

    df = df.melt(id_vars=config, var_name='metric', value_name='value')
    df = df.pivot_table(
        values="value", 
        index=["metric", "model"], 
        columns=["dataset", "buffer_size"], 
        aggfunc=np.sum
    )

    sort_dict = {
        'class-il/acc_mean': -2, 'task-il/acc_mean': -1, 
        'class-il-forget/avg':0, 'task-il-forget/avg':1, 
        'ER': 0, 'ER w/ FD': 0.5, 'ER w/ BFP': 1, 'ER w/ BFP-2': 1.5, 
        'DER++': 2, 'DER++ w/ FD': 2.5, 'DER++ w/ BFP': 3, 'DER++ w/ BFP-2': 3.5
    }
    df = df.sort_values(['metric', "model"], key=lambda x: x.map(sort_dict))

    return df