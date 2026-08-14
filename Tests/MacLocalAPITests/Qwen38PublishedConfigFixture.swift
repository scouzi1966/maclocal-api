import Foundation

enum Qwen38PublishedConfigFixture {
    static var mxfp8: [String: Any] { [
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "image_token_id": 248056,
        "video_token_id": 248057,
        "vision_start_token_id": 248053,
        "vision_end_token_id": 248054,
        "text_config": [
            "model_type": "qwen3_5_text",
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "intermediate_size": 17408,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "linear_num_value_heads": 48,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "vocab_size": 248320,
            "full_attention_interval": 4,
            "max_position_embeddings": 262144,
            "mtp_num_hidden_layers": 1,
            "rope_parameters": [
                "partial_rotary_factor": 0.25,
                "rope_theta": 10_000_000,
            ],
        ],
        "vision_config": [
            "model_type": "qwen3_5",
            "depth": 27,
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "out_hidden_size": 5120,
            "num_heads": 16,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 2304,
        ],
        "quantization": [
            "group_size": 32,
            "bits": 8,
            "mode": "mxfp8",
        ],
    ] }
}
