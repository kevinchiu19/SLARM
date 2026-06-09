from collections import defaultdict

import torch

from src.models.slarm import SLARM


class StreamSession:
    """
    A causal streaming inference session with KV cache management.
    """
    def __init__(self, model: SLARM, mode: str, window_size=4):
        self.model = model
        self.mode = mode
        self.aggregator_kv_cache_depth = self.model.aggregator.depth
        self.camera_head_kv_cache_depth = self.model.camera_head.trunk_depth if self.model.camera_head is not None else 0
        self.camera_head_iterations = 4 if self.model.camera_head is not None else 0
        self.window_size = window_size

        if self.mode not in ["causal", "window"]:
            raise ValueError(f"Unsupported attention mode when using kv_cache: {self.mode}")

        self.clear()

    def _clear_predictions(self):
        self.predictions = dict()
        named_keys = ['gs_params', 'pred_feat', 'sky_token','affine_tokens', 'pred_context_depth',
                  'pred_context_camera_enc_list','pred_context_depth_conf', 'pred_context_pts3d',
                  'pred_context_pts3d_conf']
        for k in named_keys:
            self.predictions[k] = None

    def _update_predictions(self, predictions):
        for k in ['gs_params', 'pred_feat', 'sky_token','affine_tokens', 'pred_context_depth',
                  'pred_context_camera_enc_list','pred_context_depth_conf', 'pred_context_pts3d',
                  'pred_context_pts3d_conf']:

            if k not in predictions:
                continue

            pred_value = predictions[k]

            if self.predictions.get(k, None) is None:
                self.predictions[k] = pred_value
                continue

            if k == 'sky_token' or k == 'affine_tokens':
                self.predictions[k] = pred_value # TODO: now use last token
                continue

            if k == 'pred_context_camera_enc_list':
                for i in range(len(self.predictions[k])):
                    self.predictions[k][i] = torch.cat([self.predictions[k][i], pred_value[i]], dim=1)
                continue

            if k == 'gs_params':
                for key in pred_value.keys():
                    if key == 'motion_bases': # TODO: now use last token
                        self.predictions['gs_params']['motion_bases'] = pred_value['motion_bases']
                    elif key == 'affine':
                        continue
                    else:
                        current = self.predictions['gs_params'].get(key, None)
                        self.predictions['gs_params'][key] = torch.cat([current, pred_value[key]], dim=1)
            else:
                current = self.predictions.get(k, None)
                self.predictions[k] = torch.cat([current, pred_value], dim=1)

    def _clear_cache(self):
        self.aggregator_kv_cache_list = [[None, None] for _ in range(self.aggregator_kv_cache_depth)]
        self.camera_head_kv_cache_list = [[[None, None] for _ in range(self.camera_head_kv_cache_depth)] for _ in range(self.camera_head_iterations)] if self.model.camera_head is not None else None

    def _update_cache(self, aggregator_kv_cache_list, camera_head_kv_cache_list):
        if self.mode == "causal":
            self.aggregator_kv_cache_list = aggregator_kv_cache_list
            self.camera_head_kv_cache_list = camera_head_kv_cache_list
        elif self.mode == "window": # TODO
            window_size = self.window_size
            per_frame_lens = aggregator_kv_cache_list[0][0].shape[2] // window_size
            for k in range(2):
                for i in range(self.aggregator_kv_cache_depth):
                    self.aggregator_kv_cache_list[i][k] = aggregator_kv_cache_list[i][k][:, :, per_frame_lens:]
                for i in range(self.camera_head_iterations):
                    for j in range(self.camera_head_kv_cache_depth):
                        self.camera_head_kv_cache_list[i][j][k] = camera_head_kv_cache_list[i][j][k][:, :, 1:]
        else:
            raise ValueError(f"Unsupported attention mode when using kv_cache: {self.mode}")

    def _get_cache(self):
        return self.aggregator_kv_cache_list, self.camera_head_kv_cache_list

    def get_all_predictions(self):
        return self.predictions

    def get_last_prediction(self):
        last_predictions = dict()
        for k in ["pose_enc", "world_points", "world_points_conf", "depth", "depth_conf", "images"]:
            if k in self.predictions:
                last_predictions[k] = self.predictions[k][:, -1:]
        return last_predictions

    def clear(self):
        self._clear_predictions()
        self._clear_cache()

    def forward_stream(self, input_dict, device, dtype):
        aggregator_kv_cache_list, camera_head_kv_cache_list = self._get_cache()

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=dtype):
                outputs = self.model(  # gs_params, depth ...
                    input_dict,
                    stream_save=False,
                    aggregator_kv_cache_list=aggregator_kv_cache_list,
                    camera_head_kv_cache_list=camera_head_kv_cache_list,
                )

        self._update_predictions(outputs)
        self._update_cache(outputs["aggregator_kv_cache_list"], outputs["camera_head_kv_cache_list"])

        return self.get_all_predictions()
