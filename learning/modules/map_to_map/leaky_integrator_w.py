import torch
from torch import nn as nn

PROFILE = False


class LeakyIntegrator(nn.Module):

    def __init__(self, lamda=0.2, name="leakyintegrator"):
        super(LeakyIntegrator, self).__init__()
        self.name = name
        self.lamda = lamda

    def forward(self, model_state, images_w, coverages_w, add_mask, reset_mask=None):
        seq_len = len(images_w)
        map_mem_key = f"{self.name}/map_memory"
        map_cov_key = f"{self.name}/coverage_memory"
        map_memory = model_state.get(map_mem_key, default=[])
        coverage_memory = model_state.get(map_cov_key, default=[])
        assert add_mask is None or add_mask[0] is not None, "The first observation in a sequence needs to be used!"

        masked_observations_w_add = self.lamda * images_w * coverages_w

        all_maps_out_w = []
        all_coverages_out_w = []

        # Step 2: Integrate serially in the global frame
        for i in range(seq_len):
            if len(map_memory) == 0 or (reset_mask is not None and reset_mask[i]):
                new_map_w = images_w[i:i + 1]
                new_map_cov_w = coverages_w[i:i+1]

            # Allow masking of observations
            elif add_mask is None or add_mask[i]:
                # Get the current global-frame map
                map_g = map_memory[-1]
                map_cov_g = coverage_memory[-1]
                cov_w = coverages_w[i:i+1]
                obs_cov_g = masked_observations_w_add[i:i+1]

                # Add the observation into the map using a leaky integrator rule (TODO: Output lamda from model)
                new_map_cov_w = torch.clamp(map_cov_g + cov_w, 0, 1)
                new_map_w = (1 - self.lamda) * map_g + obs_cov_g + self.lamda * map_g * (1 - cov_w)
            else:
                new_map_w = map_memory[-1]
                new_map_cov_w = self.coverage_memory[-1]

            map_memory.append(new_map_w)
            coverage_memory.append(new_map_cov_w)
            all_maps_out_w.append(new_map_w)
            all_coverages_out_w.append(new_map_cov_w)

        model_state.put(map_mem_key, map_memory)
        model_state.put(map_cov_key, coverage_memory)

        all_maps_w = torch.cat(all_maps_out_w, dim=0)
        all_coverages_out_w = torch.cat(all_coverages_out_w, dim=0)

        return all_maps_w, all_coverages_out_w