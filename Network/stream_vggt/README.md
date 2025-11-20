This is more efficient re-implementation of the work 'StreamVGGT: Streaming 4D Visual Geometry Transformer'.
It should be significantly faster than the official implementation due to auto-scaling tensor and removing redundant SDPA compute with FlexAttention.
