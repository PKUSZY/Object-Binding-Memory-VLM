"""CPU test with randomly initialized tiny Qwen3-VL; no downloaded weights/GPU."""
import numpy as np
import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from bindmem.run import Capture


def main():
    torch.set_num_threads(2)
    config = Qwen3VLConfig(
        text_config=dict(vocab_size=128,hidden_size=64,intermediate_size=128,num_hidden_layers=2,
                         num_attention_heads=4,num_key_value_heads=2,head_dim=16,
                         rope_parameters={'rope_type':'default','mrope_section':[2,3,3],'mrope_interleaved':True},
                         pad_token_id=0),
        vision_config=dict(depth=2,hidden_size=32,intermediate_size=64,num_heads=4,
                           patch_size=16,temporal_patch_size=2,spatial_merge_size=2,
                           out_hidden_size=64,num_position_embeddings=16,deepstack_visual_indexes=[0,1]),
        image_token_id=120,video_token_id=121,vision_start_token_id=122,vision_end_token_id=123)
    model = Qwen3VLForConditionalGeneration(config).eval()
    inputs = dict(input_ids=torch.tensor([[1,122,120,123,2]]),attention_mask=torch.ones((1,5),dtype=torch.long),
                  pixel_values=torch.randn(4,1536),image_grid_thw=torch.tensor([[1,2,2]]),
                  mm_token_type_ids=torch.tensor([[0,0,1,0,0]]))
    capture = Capture(model,inputs,1,'all')
    with torch.inference_mode():
        model.generate(**inputs,max_new_tokens=2,do_sample=False,pad_token_id=0,eos_token_id=127)
    capture.close()
    assert len(capture.values) == 6, capture.values.keys()
    assert all(np.isfinite(v).all() for v in capture.values.values())
    assert capture.values['vision.0.study'].shape == (32,)
    assert capture.values['decoder.1.answer'].shape == (64,)
    assert not any(m._forward_hooks for m in list(model.model.visual.blocks)+list(model.model.language_model.layers))
    print('PASS: vision/context/pre-answer hooks on tiny Qwen3-VL; handles released')


if __name__ == '__main__': main()
