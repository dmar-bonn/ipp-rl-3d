import torch
import numpy as np
import ray
import os
from attention_net import AttentionNet
from worker import Worker
from parameters import *


class Runner(object):
    """Actor object to start running simulation on workers.
    Gradient computation is also executed on this object."""

    def __init__(self, metaAgentID):
        self.metaAgentID = metaAgentID
        self.device = torch.device('cuda') if USE_GPU else torch.device('cpu')
        self.localNetwork = AttentionNet(INPUT_DIM, EMBEDDING_DIM)
        self.localNetwork.to(self.device)

    def get_weights(self):
        return self.localNetwork.state_dict()

    def set_weights(self, weights):
        self.localNetwork.load_state_dict(weights)

    def singleThreadedJob(self, episodeNumber, budget_range, sample_length, num_plants):
        save_img = True if (SAVE_IMG_GAP != 0 and episodeNumber % SAVE_IMG_GAP == 0) else False
        worker = Worker(self.metaAgentID, self.localNetwork, episodeNumber, budget_range, sample_length, num_plants, self.device, save_image=save_img, greedy=False)
        worker.work(episodeNumber)

        jobResults = worker.experience
        perf_metrics = worker.perf_metrics
        return jobResults, perf_metrics

    def job(self, global_weights, episodeNumber, budget_range, sample_length=None, num_plants = None):
        print("starting episode {} on metaAgent {}".format(episodeNumber, self.metaAgentID))
        # set the local weights to the global weight values from the master network
        self.set_weights(global_weights)

        jobResults, metrics = self.singleThreadedJob(episodeNumber, budget_range, sample_length, num_plants)

        info = {
            "id": self.metaAgentID,
            "episode_number": episodeNumber,
        }

        return jobResults, metrics, info

  
@ray.remote(num_cpus=1, num_gpus=len(CUDA_DEVICE)/NUM_META_AGENT)
class RLRunner(Runner):
    def __init__(self, metaAgentID):        
        super().__init__(metaAgentID)


if __name__=='__main__':
    ray.init()
    runner = RLRunner.remote(0)
    job_id = runner.singleThreadedJob.remote(1)
    out = ray.get(job_id)
    print(out[1])
