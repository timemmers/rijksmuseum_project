# DYNAMIC DECODING PRESETS WITH PROGRESSIVE HIGHER VALUES:
generation_presets = {
    "Deterministic": {
        "temperature": 0.7,
        "top_p": 0.80,
        "top_k": 40,
        "max_new_tokens": 1700,
        "repetition_penalty": 1.05,
    },
    "Speculative": {
        "temperature": 0.9,
        "top_p": 0.92,
        "top_k": 50,
        "max_new_tokens": 2200,
        "repetition_penalty": 1.15,
    },
}

