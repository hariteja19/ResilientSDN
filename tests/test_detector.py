import random

from controller.detector import CongestionDetector

LINK = ('s1', 's2')


def stats(u):
    return {LINK: {'utilisation': u}}


def test_triggers_only_after_N_consecutive_intervals(config, clock):
    det = CongestionDetector(config, clock=clock)
    assert det.update(stats(0.9)) == []
    assert det.update(stats(0.9)) == []
    assert det.update(stats(0.9)) == [LINK]


def test_drop_below_threshold_resets_counter(config, clock):
    det = CongestionDetector(config, clock=clock)
    det.update(stats(0.9))
    det.update(stats(0.9))
    det.update(stats(0.5))
    assert det.update(stats(0.9)) == []
    assert det.update(stats(0.9)) == []
    assert det.update(stats(0.9)) == [LINK]


def test_absent_interval_resets_counter(config, clock):
    det = CongestionDetector(config, clock=clock)
    det.update(stats(0.9))
    det.update(stats(0.9))
    det.update({})  # link missing from this poll breaks the consecutive run
    assert det.update(stats(0.9)) == []
    assert det.update(stats(0.9)) == []
    assert det.update(stats(0.9)) == [LINK]


def test_exactly_threshold_is_not_high(config, clock):
    det = CongestionDetector(config, clock=clock)
    for _ in range(5):
        assert det.update(stats(0.80)) == []


def test_repeat_attack_within_window_uses_N_minus_1(config, clock):
    det = CongestionDetector(config, clock=clock)
    for _ in range(3):
        det.update(stats(0.9))
    det.update(stats(0.1))
    clock.advance(30)
    assert det.update(stats(0.9)) == []
    assert det.update(stats(0.9)) == [LINK]  # 2 intervals, not 3


def test_repeat_attack_outside_window_uses_full_N(config, clock):
    det = CongestionDetector(config, clock=clock)
    for _ in range(3):
        det.update(stats(0.9))
    det.update(stats(0.1))
    clock.advance(config.repeat_attack_window + 1)
    det.update(stats(0.9))
    assert det.update(stats(0.9)) == []


def test_attack_confidence_is_fraction_of_recent_high_intervals(config, clock):
    det = CongestionDetector(config, clock=clock)
    assert det.get_attack_confidence(LINK) == 0.0
    for u in (0.9, 0.9, 0.1, 0.9):
        det.update(stats(u))
    assert det.get_attack_confidence(LINK) == 0.75


def test_randomised_N_stays_in_configured_range(config, clock):
    config.adversarial['randomise_N'] = True
    det = CongestionDetector(config, clock=clock, rng=random.Random(1))
    seen = set()
    for _ in range(200):
        det.update(stats(0.1))
        seen.add(det.N)
    assert seen == set(range(config.adversarial['N_min'],
                             config.adversarial['N_max'] + 1))


def _feed(det, pattern):
    """pattern of utilisations; returns the indices at which LINK triggered."""
    return [i for i, u in enumerate(pattern) if det.update(stats(u)) == [LINK]]


PULSE = [0.85, 0.85, 0.65, 0.65] * 6        # S7: 4 s high / 4 s low, polled every 2 s


def test_pulse_evades_the_consecutive_trigger(config, clock):
    assert _feed(CongestionDetector(config, clock=clock), PULSE) == []


def test_duty_cycle_trigger_catches_the_pulse_on_a_high_poll(config, clock):
    config.duty_window, config.duty_fraction = 6, 0.5
    hits = _feed(CongestionDetector(config, clock=clock), PULSE)
    assert hits and all(PULSE[i] > 0.8 for i in hits)      # only fires while the link is hot
    assert hits[0] == 5                                     # first full window with 3 of 6 high


def test_duty_cycle_trigger_ignores_short_bursts(config, clock):
    """Ablation D bursts: 3 s above the threshold every 12 s -> at most 2 of 6 polls."""
    config.duty_window, config.duty_fraction = 6, 0.5
    bursty = ([0.9, 0.9] + [0.4] * 4) * 5
    assert _feed(CongestionDetector(config, clock=clock), bursty) == []


def test_duty_window_larger_than_history_is_rejected(config, clock):
    import pytest
    config.duty_window = CongestionDetector.HISTORY_LEN + 1
    with pytest.raises(ValueError):
        CongestionDetector(config, clock=clock)


def _undetected_max_hot(config, clock, length, after):
    """Largest number of hot polls (counted from index `after`) among all patterns of `length`
    polls on which the detector never fires: exhaustive check of Proposition 3."""
    import itertools
    best = 0
    for bits in itertools.product((0, 1), repeat=length):
        det = CongestionDetector(config, clock=clock)
        if not any(det.update(stats(0.9 if b else 0.5)) for b in bits):
            best = max(best, sum(bits[after:]))
    return best


def test_proposition_3_trend_detector_bound_is_exact(config, clock):
    # N = 3: (N - 1) * ceil(n / N) = 2 * 5 = 10 hot polls in 14, and no more
    import math
    assert _undetected_max_hot(config, clock, 14, 0) == 2 * math.ceil(14 / 3)


def test_proposition_3_duty_cycle_bound_is_exact(config, clock):
    import math
    config.duty_window, config.duty_fraction = 6, 0.5
    M, rho, length = 6, 0.5, 14
    after = M - 1                                 # the bound applies once the window is full
    bound_per_window = math.ceil(rho * M) - 1     # 2 hot polls per 6
    n = length - after
    # exact: (ceil(rho M) - 1) * ceil(n / M) hot polls in any n polls after the first M - 1
    assert _undetected_max_hot(config, clock, length, after) == bound_per_window * math.ceil(n / M)
