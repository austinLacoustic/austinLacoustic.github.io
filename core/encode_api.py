# core/encode_api.py
def encode_signal_to_alak(signal_f32, sample_rate, dp,
                          prune_counts=0, use_rle=False,
                          use_basei=True):
    from core.encoder import encode_basei_sparse


    payload = encode_basei_sparse(
        signal_f32,
        dp=None if dp == "raw" else dp,
        scheme="basei_sparse_v1" if use_basei else "time_intzz_v1",
        use_rle=use_rle,
        prune_counts=int(prune_counts),
        level=1,
        use_raclp=bool(use_basei),     # complex LP only in base-i mode
        use_lp=not bool(use_basei),    # standard LP only in time_intzz mode
        lp_order=4,
    )

    return {
        "format": "alak",
        "version": "beta1",
        "sample_rate": int(sample_rate),
        "original_length": int(len(signal_f32)),
        "compressed_data": payload,
        "dp": (None if dp == "raw" else dp),
    }
