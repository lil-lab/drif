import random
import torch
import ray
import torch.nn as nn
import torch.optim as optim

from learning.inputs.partial_2d_distribution import Partial2DDistribution
from learning.modules.losses.visitation_discriminator import VisitationDiscriminator

import parameters.parameter_server as P


@ray.remote(num_gpus=0.1, num_cpus=1)
class DiscriminatorTrainer():
    def __init__(self, max_buffer_size=512, train_every_n=32, batch_size=32, steps_per_loop=8):
        self.replay_buffer = []
        self.max_buffer_size = max_buffer_size
        self.train_every_n = train_every_n
        self.steps_per_loop = steps_per_loop
        self.batch_size = batch_size
        self.push_step = 0
        self.discriminator_model = VisitationDiscriminator()
        self.optimizer = optim.Adam(self.discriminator_model.parameters(), lr=0.001, weight_decay=1e-6)

    def _train_once(self):
        data_device = self.replay_buffer[0][0].device
        self.discriminator_model = self.discriminator_model.to(data_device)
        total_score = 0
        for _ in range(self.steps_per_loop):
            bs = min(self.batch_size, len(self.replay_buffer))
            batch = random.sample(self.replay_buffer, bs)
            pred_batch, label_batch = list(zip(*batch))
            #print(len(pred_batch), len(label_batch))
            #print(pred_batch[0].inner_distribution.shape, label_batch[0].inner_distribution.shape)
            pred_batch = Partial2DDistribution.cat(pred_batch)
            label_batch = Partial2DDistribution.cat(label_batch)

            self.optimizer.zero_grad()
            scores = self.discriminator_model.calc_domain_loss(pred_batch, label_batch)
            score = scores.mean()
            loss = -score
            loss.backward()
            self.optimizer.step()
            self.discriminator_model.clip_weights()

            total_score += score.item()
        print(f"Discriminator score: {total_score}")

    def push_example_and_train(self, pred_dist, label_dist):
        self.replay_buffer.append((pred_dist, label_dist))
        if len(self.replay_buffer) >= self.max_buffer_size:
            self.replay_buffer = self.replay_buffer[1:]
        self.push_step += 1
        if self.push_step % self.train_every_n == 0:
            print(f"Discriminator actor step: {self.push_step}. Training once.")
            self._train_once()
            print(f"Discriminator training completed")

    def get_latest_parameters(self):
        return self.discriminator_model.state_dict()


discriminator_trainer_actor = None


def get_singleton_discriminator_trainer():
    global discriminator_trainer_actor
    ray.init(ignore_reinit_error=True,
             local_mode=P.get("Setup::local_ray"))
    if not discriminator_trainer_actor:
        discriminator_trainer_actor = DiscriminatorTrainer.remote()
    return discriminator_trainer_actor


class DistributedVisitationDiscriminator():

    def __init__(self, discriminator_model, run_name="", update_weights_every_n=16):
        super(DistributedVisitationDiscriminator, self).__init__()
        self.discriminator_model = discriminator_model
        self.discriminator_trainer = get_singleton_discriminator_trainer()
        self.update_weights_every_n = update_weights_every_n
        self.step = 0

    def calc_domain_loss(self, pred_dist, label_dist, eval):
        if not eval:
            # Receive latest parameters if available and freeze them
            self.step += 1
            if self.step % self.update_weights_every_n == 0:
                print(f"Discriminator step: {self.step} - updating parameters")
                m_state = ray.get(self.discriminator_trainer.get_latest_parameters.remote())
                self.discriminator_model.load_state_dict(m_state)
                for param in self.discriminator_model.parameters():
                    param.requires_grad = False

            # Broadcast distributions to discriminator trainer
            with torch.no_grad():
                pdst = Partial2DDistribution(pred_dist.inner_distribution.clone(), pred_dist.outer_prob_mass.clone())
                ldst = Partial2DDistribution(label_dist.inner_distribution.clone(), label_dist.outer_prob_mass.clone())
                self.discriminator_trainer.push_example_and_train.remote(pdst, ldst)

        # Run discriminator and get a score.
        domain_loss = self.discriminator_model.calc_domain_loss(pred_dist, label_dist)

        return domain_loss

