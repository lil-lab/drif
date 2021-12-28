import torch
from torch import nn as nn

PROFILE = False


class SegmentationMaskAccumulator(nn.Module):

    def __init__(self, name="mask_accumulator"):
        super(SegmentationMaskAccumulator, self).__init__()
        self.name = name

    def forward(self, model_state, masks_w, coverages_w, add_indicator, reset_indicator=None):
        seq_len = len(masks_w)
        map_mem_key = f"{self.name}/map_memory"
        map_cov_key = f"{self.name}/coverage_memory"
        map_memory = model_state.get(map_mem_key, default=[])
        coverage_memory = model_state.get(map_cov_key, default=[])

        assert add_indicator is None or add_indicator[0] is not None, "The first observation in a sequence needs to be used!"

        masked_observations_w_add = masks_w * coverages_w

        all_maps_out_w = []
        all_coverages_out_w = []

        # Step 2: Integrate serially in the global frame
        for i in range(seq_len):
            if len(map_memory) == 0 or (reset_indicator is not None and reset_indicator[i]):
                new_map_w = masks_w[i:i + 1]
                new_map_cov_w = coverages_w[i:i+1]

            # Allow masking of observations
            elif add_indicator is None or add_indicator[i]:
                # Get the current global-frame map
                map_g = map_memory[-1]
                map_cov_g = coverage_memory[-1]
                cov_w = coverages_w[i:i+1]
                obs_masked_observations_g = masked_observations_w_add[i:i+1]

                # Add the observation into the map using spatial max
                # Merge map coverage using a max function:
                new_map_cov_w = torch.clamp(map_cov_g + cov_w, 0, 1)
                new_map_w = torch.max(map_g, obs_masked_observations_g)
            else:
                new_map_cov_w = self.coverage_memory[-1]
                new_map_w = map_memory[-1]

            map_memory.append(new_map_w)
            coverage_memory.append(new_map_cov_w)
            all_maps_out_w.append(new_map_w)
            all_coverages_out_w.append(new_map_cov_w)

        model_state.put(map_mem_key, map_memory)
        model_state.put(map_cov_key, coverage_memory)

        all_maps_w = torch.cat(all_maps_out_w, dim=0)
        all_coverages_out_w = torch.cat(all_coverages_out_w, dim=0)

        return all_maps_w, all_coverages_out_w