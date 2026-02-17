from loadgen.metrics import percentile, summarize_latencies, tokens_per_second


def test_percentile_linear_interp():
    vals = [0.0, 10.0]
    assert percentile(vals, 0) == 0.0
    assert percentile(vals, 100) == 10.0
    assert percentile(vals, 50) == 5.0


def test_summarize_latencies():
    s = summarize_latencies([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(s.mean_s - 3.0) < 1e-12
    assert s.min_s == 1.0
    assert s.max_s == 5.0
    assert s.p50_s is not None and 2.9 < s.p50_s < 3.1


def test_tokens_per_second():
    assert tokens_per_second(100, 10.0) == 10.0
    assert tokens_per_second(0, 10.0) is None
    assert tokens_per_second(100, 0.0) is None
