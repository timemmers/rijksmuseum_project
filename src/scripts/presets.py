# DYNAMIC DECODING PRESETS WITH PROGRESSIVE HIGHER VALUES:
generation_presets = {
    "Deterministic": {
        "temperature": 0.4,
        "top_p": 0.85,
        "top_k": 40,
        "max_new_tokens": 2000,
        "repetition_penalty": 1.08,
    },
    "Speculative": {
        "temperature": 0.7,
        "top_p": 0.92,
        "top_k": 60,
        "max_new_tokens": 2500,
        "repetition_penalty": 1.10,
    },
}