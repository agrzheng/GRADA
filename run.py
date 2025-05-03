import os
from imaplib import Flags


def run(test_params):

    log_file, log_name = get_log_name(test_params)

    cmd = f"nohup python3 -u main.py \
        --eval_model_code {test_params['eval_model_code']}\
        --eval_dataset {test_params['eval_dataset']}\
        --split {test_params['split']}\
        --query_results_dir {test_params['query_results_dir']}\
        --model_name {test_params['model_name']}\
        --top_k {test_params['top_k']}\
        --use_truth {test_params['use_truth']}\
        --gpu_id {test_params['gpu_id']}\
        --attack_method {test_params['attack_method']}\
        --adv_per_query {test_params['adv_per_query']}\
        --score_function {test_params['score_function']}\
        --repeat_times {test_params['repeat_times']}\
        --M {test_params['M']}\
        --seed {test_params['seed']}\
        --defence {test_params['defence']}\
        --name {log_name}\
        > {log_file} "
        
    exit_code = os.system(cmd)
    if exit_code != 0:
        print(f"Error: {file} exited with code {exit_code}")


def get_log_name(test_params):
    # Generate a log file name
    os.makedirs(f"logs/{test_params['query_results_dir']}_logs/{test_params['model_name']}/{test_params['eval_dataset']}/{test_params['attack_method']}", exist_ok=True)

    if test_params['use_truth']:
        log_name = f"{test_params['eval_dataset']}-{test_params['eval_model_code']}-{test_params['model_name']}-Truth--M{test_params['M']}x{test_params['repeat_times']}-{test_params['defence']}-{test_params['seed']}"
    else:
        log_name = f"{test_params['eval_dataset']}-{test_params['eval_model_code']}-{test_params['model_name']}-Top{test_params['top_k']}--M{test_params['M']}x{test_params['repeat_times']}"
    
    if test_params['attack_method'] != None:
        log_name += f"-adv-{test_params['attack_method']}-{test_params['score_function']}-{test_params['adv_per_query']}-{test_params['top_k']}-{test_params['defence']}-{test_params['seed']}"

    if test_params['note'] != None:
        log_name = test_params['note']
    
    return f"logs/{test_params['query_results_dir']}_logs/{test_params['model_name']}/{test_params['eval_dataset']}/{test_params['attack_method']}/{log_name}.txt", log_name



test_params = {
    # beir_info
    'eval_model_code': "contriever",
    'eval_dataset': "nq",
    'split': "test",
    'query_results_dir': 'main',

    # LLM setting
    'model_name': 'gpt3.5',
    'use_truth': False,
    'top_k': 10,
    'gpu_id': 0,

    # attack
    'attack_method': 'hotflip',
    'adv_per_query': 1,
    'score_function': 'dot',
    'repeat_times': 10,
    'M': 10,
    'seed': 12,
    'defence': "textranking_cos",
    'run_times': 0,

    'note': None
}

for seeds in range(0,3):
    test_params['seed'] = seeds
    for dataset in ['nq','hotpotqa', 'msmarco']:
        test_params['eval_dataset'] = dataset

        #Test for no attacks
        for attack in ["no"]:
            test_params['attack_method'] = None
            test_params["use_truth"] = True
            for defence in ['no', 'EBD', 'BM25','HRSIM']:
                test_params['defence'] = defence
                run(test_params)

        #Test for attacks
        for attack in ['hotflip','LM_targeted','PIA','phantom']:
            test_params["use_truth"] = False
            test_params['attack_method'] = attack
            for defence in ['no', 'EBD', 'BM25','HRSIM']:
                test_params['defence'] = defence
                run(test_params)

