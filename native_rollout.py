"""Current native bit-run codec and optional minibatch-only CUDA expansion."""
import math

import numpy as np
import torch
from sb3_contrib.common.maskable.buffers import MaskableRolloutBufferSamples

from compact_rollout import CompactMaskableRolloutBuffer
from native_codec import NativeRunCodec


class NativeRunRolloutBuffer(CompactMaskableRolloutBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        scratch_bytes = math.prod(self.frame_shape) * 4 + 1
        self._check_capacity(self.total_storage_nbytes + scratch_bytes)
        self.native_codec = NativeRunCodec(self.frame_shape)
        self._array_nbytes += self.native_codec.scratch.nbytes

    def _encode_frame(self, frame):
        return self.native_codec.encode(frame)

    def _decode_frame(self, payload):
        return self.native_codec.decode(payload)


class NativeGPURunRolloutBuffer(NativeRunRolloutBuffer):
    """Expand integer bit patterns, never quantize or cast floating values."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu_decoded_minibatches = 0

    def _get_samples(self, batch_inds, env=None):
        steps = batch_inds % self.buffer_size
        envs = batch_inds // self.buffer_size
        ids = self._frame_ids[steps, envs].ravel()
        payloads = [self._frames[int(frame_id)] for frame_id in ids]
        if self.device.type != 'cuda' or any(payload[0] != 1 for payload in payloads):
            return super()._get_samples(batch_inds, env)
        runs = np.concatenate([np.frombuffer(payload, dtype='<u4', offset=1).reshape(-1, 2)
                               for payload in payloads])
        counts = torch.as_tensor(runs[:, 0].astype(np.int64), device=self.device)
        bits = torch.as_tensor(runs[:, 1].copy().view(np.int32), device=self.device)
        expected = len(batch_inds) * math.prod(self.obs_shape)
        observations = torch.repeat_interleave(bits, counts, output_size=expected).view(
            torch.float32).reshape(len(batch_inds), *self.obs_shape)
        self.gpu_decoded_minibatches += 1
        scalars = (
            self.actions[steps, envs], self.values[steps, envs].flatten(),
            self.log_probs[steps, envs].flatten(), self.advantages[steps, envs].flatten(),
            self.returns[steps, envs].flatten(),
            self.action_masks[steps, envs].astype(np.float32).reshape(-1, self.mask_dims),
        )
        return MaskableRolloutBufferSamples(observations, *map(self.to_torch, scalars))
