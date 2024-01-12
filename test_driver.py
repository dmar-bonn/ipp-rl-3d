import copy
import csv
import os
import ray
import torch
import time
from multiprocessing import Pool
import numpy as np
import time
from attention_net import AttentionNet
from runner import Runner
from test_worker import WorkerTest
from test_parameters import *


def run_test():
    time0 = time.time()
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    device = torch.device('cuda') if USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if USE_GPU else torch.device('cpu')
    global_network = AttentionNet(INPUT_DIM, EMBEDDING_DIM).to(device)

    if not USE_GPU_GLOBAL:
        checkpoint = torch.load(f'{model_path}/checkpoint.pth', map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(f'{model_path}/checkpoint.pth')
    global_network.load_state_dict(checkpoint['model'])

    print(f'Loading model: {FOLDER_NAME}...')
    print(f'Total budget range: {BUDGET_RANGE}')

    # init meta agents
    meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]
    weights = global_network.to(local_device).state_dict() if device != local_device else global_network.state_dict()
    curr_test = 1
    metric_name = ['remain_budget', 'success_rate', 'RMSE', 'delta_cov_trace', 'F1Score', 'cov_trace', 'node_utils', 'entropy', 'detection_rate']

    perf_metrics = {}
    for n in metric_name:
        perf_metrics[n] = []
    cov_trace_list = []
    time_list = []
    episode_number_list = []
    budget_history = []
    obj_history = []
    obj2_history = []

    try:
        while True:
            jobList = []
            for i, meta_agent in enumerate(meta_agents):
                jobList.append(meta_agent.job.remote(weights, curr_test, budget_range=BUDGET_RANGE, sample_length=SAMPLE_LENGTH))
                curr_test += 1
            done_id, jobList = ray.wait(jobList, num_returns=NUM_META_AGENT)
            done_jobs = ray.get(done_id)

            for job in done_jobs:
                metrics, info = job
                episode_number_list.append(info['episode_number'])
                cov_trace_list.append(metrics['cov_trace'])
                time_list.append(metrics['planning_time'])
                for n in metric_name:
                    perf_metrics[n].append(metrics[n])
                budget_history += metrics['budget_history']
                obj_history += metrics['obj_history']
                obj2_history += metrics['obj2_history']


            if curr_test > NUM_TEST:
                print('#Test sample:', NUM_SAMPLE_TEST, '|#Total test:', NUM_TEST, '|Budget range:', BUDGET_RANGE, '|Sample size:', SAMPLE_SIZE, '|K size:', K_SIZE)
                print('Avg time per test:', (time.time()-time0)/NUM_TEST)
                perf_data = []
                for n in metric_name:
                    perf_data.append(np.nanmean(perf_metrics[n]))
                for i in range(len(metric_name)):
                    print(metric_name[i], ':\t', perf_data[i])
                
                idx = np.array(episode_number_list).argsort()
                cov_trace_list = np.array(cov_trace_list)[idx]
                time_list = np.array(time_list)[idx]

                if SAVE_TRAJECTORY_HISTORY:
                    idx = np.array(budget_history).argsort()
                    budget_history = np.array(budget_history)[idx]
                    obj_history = np.array(obj_history)[idx]
                    obj2_history = np.array(obj2_history)[idx]

                break

        Budget = int(perf_data[0])+1
        if SAVE_CSV_RESULT:
            if TRAJECTORY_SAMPLING:
                csv_filename = f'result/CSV/Budget_'+str(Budget)+'_ts_'+str(PLAN_STEP)+'_'+str(NUM_SAMPLE_TEST)+'_'+str(SAMPLE_SIZE)+'_'+str(K_SIZE)+'_results.csv'
                csv_filename3 = f'result/CSV3/Budget_'+str(Budget)+'_ts_'+str(PLAN_STEP)+'_'+str(NUM_SAMPLE_TEST)+'_'+str(SAMPLE_SIZE)+'_'+str(K_SIZE)+'_planning_time.csv'
            else:
                csv_filename = f'result/CSV/Budget_'+str(Budget)+'_greedy'+'_'+str(SAMPLE_SIZE)+'_'+str(K_SIZE)+'_results.csv'
                csv_filename3 = f'result/CSV3/Budget_'+str(Budget)+'_greedy'+'_'+str(SAMPLE_SIZE)+'_'+str(K_SIZE)+'_planning_time.csv'
            csv_data = [cov_trace_list]
            csv_data3 = [time_list]
            with open(csv_filename, 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(csv_data)
            with open(csv_filename3, 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(csv_data3)

        if SAVE_TRAJECTORY_HISTORY:
            if TRAJECTORY_SAMPLING:
                csv_filename2 = f'result/CSV2/Budget_'+str(Budget)+'_ts_'+str(PLAN_STEP)+'_'+str(NUM_SAMPLE_TEST)+'_'+str(SAMPLE_SIZE)+'_'+str(K_SIZE)+'_trajectory_result.csv'
            else:
                csv_filename2 = f'result/CSV2/Budget_'+str(Budget)+'_greedy_'+'_'+str(SAMPLE_SIZE)+'_'+str(K_SIZE)+'_trajectory_result.csv'
            new_file = False if os.path.exists(csv_filename2) else True
            field_names = ['budget','obj','obj2']
            with open(csv_filename2, 'a') as csvfile:
                writer = csv.writer(csvfile)
                if new_file:
                    writer.writerow(field_names)
                csv_data = np.concatenate((budget_history.reshape(-1,1), obj_history.reshape(-1,1), obj2_history.reshape(-1,1)), axis=-1)
                writer.writerows(csv_data)

    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)


@ray.remote(num_cpus=8/NUM_META_AGENT, num_gpus=NUM_GPU/NUM_META_AGENT)
class RLRunner(Runner):
    def __init__(self, metaAgentID):
        super().__init__(metaAgentID)

    def singleThreadedJob(self, episodeNumber, budget_range, sample_length):
        save_img = True if episodeNumber % SAVE_IMG_GAP == 0 else False
        np.random.seed(SEED + 100 * episodeNumber)
        worker = WorkerTest(self.metaAgentID, self.localNetwork, episodeNumber, budget_range, SAMPLE_SIZE, sample_length, 15, self.device, save_image=save_img, greedy=False, seed=SEED + 100 * episodeNumber) 
        worker.work(episodeNumber, 0)
        perf_metrics = worker.perf_metrics
        return perf_metrics

    def multiThreadedJob(self, episodeNumber, budget_range, sample_length):
        save_img = True if (SAVE_IMG_GAP != 0 and episodeNumber % SAVE_IMG_GAP == 0) else False
        np.random.seed(SEED + 100 * episodeNumber)
        worker = WorkerTest(self.metaAgentID, self.localNetwork, episodeNumber, budget_range, sample_length, self.device, save_image=save_img, greedy=False, seed=SEED + 100 * episodeNumber)
        subworkers = [copy.deepcopy(worker) for _ in range(NUM_SAMPLE_TEST)]
        p = Pool(processes=NUM_SAMPLE_TEST)
        results = []
        for testID, subw in enumerate(subworkers):
            results.append(p.apply_async(subw.work, args=(episodeNumber, testID+1)))
        p.close()
        p.join()
        all_results = []
        best_score = np.inf
        perf_metrics = None
        for res in results:
            metric = res.get()
            all_results.append(metric)
            if metric['cov_trace'] < best_score: # TODO
                perf_metrics = metric
                best_score = metric['cov_trace']
        return perf_metrics

    def job(self, global_weights, episodeNumber, budget_range, sample_length=None):
        self.set_weights(global_weights)
        metrics = self.singleThreadedJob(episodeNumber, budget_range, sample_length)

        info = {
            "id": self.metaAgentID,
            "episode_number": episodeNumber,
        }

        return metrics, info


if __name__ == '__main__':
    ray.init()
    run_test()
