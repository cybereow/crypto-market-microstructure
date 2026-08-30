import pandas as pd

from scripts.download_l2_obi import process_bookticker_chunk, _resample_chunk


def _raw_chunk(rows):
    """rows: list of (transaction_time_ms, bid_price, bid_qty, ask_price, ask_qty)."""
    return pd.DataFrame(rows, columns=[
        'update_id', 'best_bid_price', 'best_bid_qty', 'best_ask_price',
        'best_ask_qty', 'transaction_time', 'event_time',
    ])


def test_process_bookticker_chunk_computes_mid_and_renames_header():
    raw = pd.DataFrame([
        [1, 100.0, 5.0, 102.0, 3.0, 1_700_000_000_000, 1_700_000_000_001],
        [2, 101.0, 4.0, 103.0, 2.0, 1_700_000_000_500, 1_700_000_000_501],
    ], columns=['a', 'b', 'c', 'd', 'e', 'f', 'g'])  # no header, generic names

    out = process_bookticker_chunk(raw)

    assert list(out.columns) == ['timestamp', 'bid_qty', 'ask_qty', 'bid_price', 'ask_price', 'mid']
    assert out['mid'].tolist() == [101.0, 102.0]
    assert out['bid_qty'].tolist() == [5.0, 4.0]
    assert out['ask_qty'].tolist() == [3.0, 2.0]
    assert out['bid_price'].tolist() == [100.0, 101.0]
    assert out['ask_price'].tolist() == [102.0, 103.0]


def test_resample_chunk_sums_qty_and_builds_ohlc_from_mid():
    """Within one 10-second bucket: qty sums, and mid becomes a proper OHLC
    bar (first/max/min/last), not an average.
    """
    raw = _raw_chunk([
        (1_705_312_800_000, 100.0, 1.0, 102.0, 1.0, 1_705_312_800_000, 0),  # mid=101, t=00:00:00
        (0, 103.0, 2.0, 105.0, 2.0, 1_705_312_802_000, 0),                  # mid=104, t=00:00:02
        (0, 98.0, 3.0, 100.0, 3.0, 1_705_312_805_000, 0),                   # mid=99,  t=00:00:05
        (0, 101.0, 4.0, 103.0, 4.0, 1_705_312_807_000, 0),                  # mid=102, t=00:00:07
    ])
    processed = process_bookticker_chunk(raw)
    processed.set_index('timestamp', inplace=True)

    out = _resample_chunk(processed, '10s')

    assert len(out) == 1
    row = out.iloc[0]
    assert row['bid_qty_sum'] == 10.0   # 1+2+3+4
    assert row['ask_qty_sum'] == 10.0
    assert row['mid_first'] == 101.0
    assert row['mid_max'] == 104.0
    assert row['mid_min'] == 99.0
    assert row['mid_last'] == 102.0
    # Real last-observed quotes, NOT derived from mid -- the tick at t=07 had
    # bid=101, ask=103 (mid=102), so the bar's closing quotes come from there.
    assert row['bid_price_last'] == 101.0
    assert row['ask_price_last'] == 103.0


def test_cross_chunk_combine_reproduces_true_ohlc_when_bucket_splits_chunks():
    """A single 10s bucket's ticks split across two chunks (the real
    scenario at a 100k-row chunk boundary): re-aggregating the two
    per-chunk resampled frames with the same sum/first/max/min/last combine
    used in download_and_process_l2/main() must reproduce the SAME result
    as resampling all ticks in one pass -- this is the correctness
    invariant the whole chunked-processing approach depends on.
    """
    all_ticks = _raw_chunk([
        (1_705_312_800_000, 100.0, 1.0, 102.0, 1.0, 1_705_312_800_000, 0),  # mid=101 (first)
        (0, 103.0, 2.0, 105.0, 2.0, 1_705_312_802_000, 0),                  # mid=104 (max)
        (0, 98.0, 3.0, 100.0, 3.0, 1_705_312_805_000, 0),                   # mid=99  (min)
        (0, 101.0, 4.0, 103.0, 4.0, 1_705_312_807_000, 0),                  # mid=102 (last)
    ])
    processed = process_bookticker_chunk(all_ticks)
    processed.set_index('timestamp', inplace=True)
    single_pass = _resample_chunk(processed, '10s')

    # Same ticks, split 2/2 across two "chunks" processed separately.
    chunk_a = process_bookticker_chunk(all_ticks.iloc[:2])
    chunk_a.set_index('timestamp', inplace=True)
    chunk_b = process_bookticker_chunk(all_ticks.iloc[2:])
    chunk_b.set_index('timestamp', inplace=True)
    resampled_a = _resample_chunk(chunk_a, '10s')
    resampled_b = _resample_chunk(chunk_b, '10s')

    combined = pd.concat([resampled_a, resampled_b])
    recombined = combined.groupby(combined.index).agg({
        'bid_qty_sum': 'sum', 'ask_qty_sum': 'sum',
        'mid_first': 'first', 'mid_max': 'max', 'mid_min': 'min', 'mid_last': 'last',
        'bid_price_last': 'last', 'ask_price_last': 'last',
    })

    pd.testing.assert_frame_equal(recombined, single_pass, check_freq=False)
