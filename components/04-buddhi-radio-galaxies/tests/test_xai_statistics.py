"""Synthetic scalar records and analytical statistics; no research data access."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from src.xai import statistics as stats


def record(index, label, value, *, predicted=None, metric="x"):
    return stats.image_record(index, label, predicted, {metric: value})


def controls(value=.1):
    return stats.aggregate_repeats({seed: value for seed in range(1000, 1020)}, "random_deletion")


def primary_records():
    return tuple(stats.image_record(i, i % 2, i % 2,
                 {stats.CAM: .3+i*.01, stats.RANDOM: controls(.1), stats.BRIGHTNESS: .2})
                 for i in range(6))


def test_constants_match_frozen_configuration():
    config = yaml.safe_load(Path("config/xai_phase3b_v1.0.0.yaml").read_text())["statistics"]
    assert stats.REPLICATES == config["bootstrap"]["replicates"] == config["permutation"]["replicates"] == 10000
    assert stats.BOOTSTRAP_SEED == config["bootstrap"]["seed"] == 42
    assert stats.PERMUTATION_SEED == config["permutation"]["seed"] == 43
    assert stats.PRIMARY_FAMILY == tuple(config["inferential_family"])
    assert stats.ALPHA == config["correction"]["familywise_alpha"] == .05
    assert config["bootstrap"]["interval_percentiles"] == [2.5, 97.5]


def test_descriptive_arithmetic_quartiles_sd_and_macro():
    records = (record(0, 0, 1, predicted=0), record(1, 0, 3, predicted=1),
               record(2, 1, 10, predicted=1))
    result = stats.summarize(records, "x")
    overall = result.overall
    assert (overall.total, overall.available, overall.unavailable) == (3, 3, 0)
    assert overall.mean == pytest.approx(14/3)
    assert (overall.q1, overall.median, overall.q3, overall.iqr) == (2, 3, 6.5, 4.5)
    assert overall.sample_standard_deviation == pytest.approx(np.sqrt(67/3))
    assert [g.mean for g in result.by_class] == [2, 10]
    assert result.equal_class_macro_mean == 6
    assert result.sample_weighted_overall_mean == overall.mean
    assert result.correctly_classified.mean == 5.5
    assert result.misclassified.mean == 3
    assert result.unknown_prediction.total == 0


def test_metric_specific_denominators_and_multiple_failure_reasons():
    records = (stats.image_record(0, 0, 0, {"x": 1, "y": None}),
               stats.image_record(1, 1, None, {"x": float("nan"), "y": 5}),
               stats.image_record(2, 1, 0, {"x": stats.Observation(None, "unavailable", ("degenerate", "support"))}),
               stats.image_record(3, 0, 1, {}))
    x, y = stats.summarize(records, "x"), stats.summarize(records, "y")
    assert x.overall.available == y.overall.available == 1
    assert x.overall.unavailable == 3
    assert dict(x.overall.unavailable_reasons) == {"nonfinite": 1, "degenerate": 1, "support": 1, "missing_metric": 1}
    assert dict(y.overall.unavailable_reasons) == {"missing": 1, "missing_metric": 2}
    assert x.equal_class_macro_mean is None
    assert x.macro_status == "missing_class"
    assert y.unknown_prediction.available == 1


def test_empty_all_missing_and_singleton_subgroups():
    empty = stats.summarize((), "x")
    assert empty.overall.total == 0 and empty.overall.mean is None
    assert empty.overall.status == "unavailable"
    missing = stats.summarize((record(0, 0, None), record(1, 1, None)), "x")
    assert missing.overall.available == 0 and missing.overall.unavailable == 2
    singleton = stats.summarize((record(0, 0, 7),), "x")
    assert singleton.overall.mean == singleton.overall.median == 7
    assert singleton.overall.iqr == 0
    assert singleton.overall.sample_standard_deviation is None
    assert singleton.overall.warnings == ("fewer_than_two_available_images",)


def test_random_controls_averaged_once_per_image_not_twenty_units():
    values = {seed: float(seed-1000) for seed in range(1000, 1020)}
    aggregate = stats.aggregate_repeats(values, "random_deletion")
    assert aggregate.value == 9.5 and aggregate.available_repeats == 20
    result = stats.summarize((record(0, 0, aggregate), record(1, 1, controls(2))), "x")
    assert result.overall.available == 2
    assert result.overall.repeat_total == result.overall.repeat_available == 40
    assert result.overall.mean == 5.75


@pytest.mark.parametrize("kind,setting,seeds", [("noise", .005, range(2000, 2010)),
                                             ("noise", .01, range(2000, 2010)),
                                             ("randomization", 1, range(3000, 3005)),
                                             ("randomization", 6, range(3000, 3005))])
def test_noise_and_randomization_image_setting_averages(kind, setting, seeds):
    values = {seed: float(seed-seeds.start) for seed in seeds}
    aggregate = stats.aggregate_repeats(values, kind, setting)
    assert aggregate.value == np.mean(tuple(values.values()))
    assert aggregate.setting == setting and aggregate.repeat_kind == kind
    assert tuple(seed for seed, _ in aggregate.repeats) == tuple(seeds)
    assert stats.summarize((record(0, 0, aggregate),), "x").overall.available == 1


def test_partial_descriptive_repeat_coverage_is_explicit():
    values = {seed: 2 for seed in range(2000, 2010)}
    values[2000] = float("inf")
    del values[2001]
    aggregate = stats.aggregate_repeats(values, "noise", .005)
    assert aggregate.status == "available" and aggregate.value == 2
    assert aggregate.available_repeats == 8 and len(aggregate.repeats) == 10
    summary = stats.summarize((record(0, 0, aggregate),), "x").overall
    assert summary.available == 1 and summary.repeat_available == 8
    assert summary.repeat_unavailable == 2
    assert dict(summary.repeat_unavailable_reasons) == {"nonfinite": 1, "missing_repeat": 1}


def test_incomplete_random_control_not_partially_averaged():
    values = {seed: .1 for seed in range(1000, 1019)}
    aggregate = stats.aggregate_repeats(values, "random_deletion")
    assert aggregate.value is None and aggregate.reasons == ("incomplete_random_controls",)
    assert aggregate.available_repeats == 19
    empty = stats.aggregate_repeats({}, "randomization", 2)
    assert empty.reasons == ("all_repeats_unavailable",)


def test_settings_are_not_pooled_and_raw_repeats_retained():
    low = stats.aggregate_repeats({s: 1 for s in range(2000, 2010)}, "noise", .005)
    high = stats.aggregate_repeats({s: 3 for s in range(2000, 2010)}, "noise", .01)
    r = stats.image_record(0, 0, 0, {"noise_005": low, "noise_01": high})
    assert stats.summarize((r,), "noise_005").overall.mean == 1
    assert stats.summarize((r,), "noise_01").overall.mean == 3
    assert dict(r.metrics)["noise_005"].repeats == low.repeats


@pytest.mark.parametrize("kind,setting,values", [("noise", .02, {}), ("noise", None, {}),
                                               ("randomization", 0, {}), ("randomization", 1., {}),
                                               ("random_deletion", None, {999: 0}),
                                               ("noise", .005, {True: 0})])
def test_invalid_repeat_definitions(kind, setting, values):
    with pytest.raises(ValueError):
        stats.aggregate_repeats(values, kind, setting)


def test_forged_or_nested_aggregates_rejected():
    aggregate = controls()
    with pytest.raises(ValueError):
        record(0, 0, replace(aggregate, value=999))
    with pytest.raises(ValueError):
        stats.aggregate_repeats({1000: aggregate}, "random_deletion")
    with pytest.raises(ValueError):
        record(0, 0, replace(aggregate, repeats=aggregate.repeats + aggregate.repeats[:1]))


def test_primary_paired_contrasts_and_negative_values():
    records = primary_records()
    derived = stats.add_primary_contrasts(records)
    for original, output in zip(records, derived):
        values = dict(output.metrics)
        assert values[stats.PRIMARY_FAMILY[0]].value == pytest.approx(dict(original.metrics)[stats.CAM].value-.1)
        assert values[stats.PRIMARY_FAMILY[1]].value == pytest.approx(dict(original.metrics)[stats.CAM].value-.2)
        assert values[stats.RANDOM].repeats == dict(original.metrics)[stats.RANDOM].repeats
    negative = stats.image_record(0, 0, 0, {stats.CAM: -.2, stats.RANDOM: controls(.1), stats.BRIGHTNESS: .2})
    contrasts = dict(stats.add_primary_contrasts((negative,))[0].metrics)
    assert contrasts[stats.PRIMARY_FAMILY[0]].value == pytest.approx(-.3)
    assert contrasts[stats.PRIMARY_FAMILY[1]].value == pytest.approx(-.4)


def test_incomplete_pairs_have_independent_denominators():
    records = (stats.image_record(0, 0, 0, {stats.CAM: .3, stats.RANDOM: controls()}),
               stats.image_record(1, 1, 1, {stats.CAM: .4, stats.BRIGHTNESS: .1}),
               stats.image_record(2, 1, 1, {stats.CAM: stats.observation(reason="missing_deletion_step"),
                                                       stats.RANDOM: controls(), stats.BRIGHTNESS: .2}))
    derived = stats.add_primary_contrasts(records)
    assert stats.summarize(derived, stats.PRIMARY_FAMILY[0]).overall.available == 1
    assert stats.summarize(derived, stats.PRIMARY_FAMILY[1]).overall.available == 1
    assert "cam_probability_aopc30:missing_deletion_step" in dict(derived[2].metrics)[stats.PRIMARY_FAMILY[0]].reasons
    scalar = stats.image_record(3, 0, 0, {stats.CAM: .3, stats.RANDOM: .1, stats.BRIGHTNESS: .2})
    assert "unverified_random_control_aggregation" in dict(stats.add_primary_contrasts((scalar,))[0].metrics)[stats.PRIMARY_FAMILY[0]].reasons
    with pytest.raises(ValueError):
        stats.add_primary_contrasts(derived)


def test_bootstrap_determinism_pairing_stratification_and_rng():
    records = tuple(stats.image_record(i, i % 2, 0, {"x": float(i), "y": float(i)+7}) for i in range(6))
    state = deepcopy(np.random.get_state())
    first = stats.bootstrap_paired(records, ("x", "y"))
    second = stats.bootstrap_paired(records[::-1], ("x", "y"))
    assert first == second
    assert first.replicates == 10000 and first.seed == 42
    assert len(first.macro_replicates) == len(first.overall_replicates) == 10000
    np.testing.assert_allclose(np.diff(first.macro_replicates, axis=1), 7, atol=1e-14)
    np.testing.assert_allclose(np.diff(first.overall_replicates, axis=1), 7, atol=1e-14)
    labels = np.array([r.ground_truth_label for r in records])
    indices = stats._bootstrap_indices(labels)
    assert indices.shape == (10000, 6)
    assert np.all((labels[indices] == 0).sum(axis=1) == 3)
    assert np.all((labels[indices] == 1).sum(axis=1) == 3)
    current = np.random.get_state()
    assert state[0] == current[0] and state[2:] == current[2:]
    np.testing.assert_array_equal(state[1], current[1])


def test_percentile_intervals_known_finite_population():
    records = (record(0, 0, 0), record(1, 0, 2), record(2, 1, 10))
    result = stats.bootstrap_paired(records, ("x",))
    assert result.coverage.class_counts == (2, 1)
    assert result.coverage.warnings == ("singleton_class_stratum",)
    assert result.macro_estimates == (5.5,)
    assert result.overall_estimates == (4.,)
    assert result.macro_intervals == ((5., 6.),)
    np.testing.assert_allclose(result.overall_intervals, ((10/3, 14/3),))
    expected = np.percentile(result.macro_replicates, [2.5, 97.5], axis=0, method="linear").T
    np.testing.assert_array_equal(result.macro_intervals, expected)


def test_joint_complete_cases_and_conditional_interval_metadata():
    records = (stats.image_record(0, 0, 0, {"x": 1, "y": 2}),
               stats.image_record(1, 1, 1, {"x": 3, "y": 4}),
               stats.image_record(2, 0, 0, {"x": 5, "y": None}))
    joint = stats.bootstrap_paired(records, ("x", "y"))
    single = stats.bootstrap_paired(records, ("x",))
    assert joint.coverage.image_ids == (0, 1)
    assert joint.coverage.conditional_on_metric_availability
    assert joint.coverage.excluded == ((2, ("y:missing",)),)
    assert single.coverage.image_ids == (0, 1, 2)
    assert not single.coverage.conditional_on_metric_availability


@pytest.mark.parametrize("records", [(), (record(0, 0, 1),), (record(0, 0, 1), record(1, 0, 2)),
                                    (record(0, 0, None), record(1, 1, None))])
def test_bootstrap_absent_classes_and_all_missing_unavailable(records):
    result = stats.bootstrap_paired(records, ("x",))
    assert result.coverage.status == "unavailable"
    assert "missing_class" in result.coverage.reasons
    assert result.macro_intervals is result.macro_estimates is None
    assert result.macro_replicates == ()


def test_two_singleton_classes_are_flagged_but_follow_frozen_rule():
    result = stats.bootstrap_paired((record(0, 0, 1), record(1, 1, 3)), ("x",))
    assert result.coverage.status == "available"
    assert result.coverage.warnings == ("singleton_class_stratum",)
    assert result.macro_intervals == ((2., 2.),)


def test_no_seed_pseudoreplication_in_bootstrap():
    records = (record(0, 0, controls(1)), record(1, 1, controls(3)))
    result = stats.bootstrap_paired(records, ("x",))
    assert result.coverage.image_ids == (0, 1)
    assert result.coverage.class_counts == (1, 1)
    assert result.macro_intervals == ((2., 2.),)
    assert all(row == (2.,) for row in result.macro_replicates)


def test_sign_flip_known_example_plus_one_and_exact_rng():
    name = stats.PRIMARY_FAMILY[0]
    records = (record(0, 0, 1, metric=name), record(1, 1, 3, metric=name))
    result = stats.primary_sign_flip(records, name)
    assert result.statistic == 2
    rng = np.random.Generator(np.random.PCG64(43))
    signs = 2*rng.integers(0, 2, (10000, 2), dtype=np.int8)-1
    expected = int(np.count_nonzero(signs[:, 0] == signs[:, 1]))
    assert result.as_or_more_extreme == expected
    assert result.p_value == (1+expected)/10001
    assert result.permutations == 10000 and result.seed == 43
    assert result == stats.primary_sign_flip(records[::-1], name)


def test_sign_flip_equal_class_macro_not_sample_weighted():
    name = stats.PRIMARY_FAMILY[0]
    records = tuple(record(i, 0 if i < 3 else 1, 1 if i < 3 else -3, metric=name) for i in range(4))
    assert stats.primary_sign_flip(records, name).statistic == pytest.approx(-1)


def test_sign_flip_zero_null_and_consistent_signal():
    name = stats.PRIMARY_FAMILY[1]
    zero = tuple(record(i, i % 2, 0, metric=name) for i in range(20))
    result = stats.primary_sign_flip(zero, name)
    assert result.p_value == 1 and result.as_or_more_extreme == 10000
    signal = tuple(record(i, i % 2, 1, metric=name) for i in range(20))
    positive = stats.primary_sign_flip(signal, name)
    negative = stats.primary_sign_flip(tuple(record(i, i % 2, -1, metric=name) for i in range(20)), name)
    assert positive.p_value == negative.p_value < .001
    assert positive.p_value >= 1/10001


def test_sign_flip_unavailable_and_extra_tests_rejected():
    name = stats.PRIMARY_FAMILY[0]
    result = stats.primary_sign_flip((record(0, 0, None, metric=name),), name)
    assert result.p_value is result.statistic is None
    with pytest.raises(ValueError, match="additional confirmatory"):
        stats.primary_sign_flip((), "stability_jaccard")


@pytest.mark.parametrize("raw,adjusted,reject", [((.01, .04), (.02, .04), (True, True)),
                                               ((.03, .04), (.06, .06), (False, False)),
                                               ((.04, .01), (.04, .02), (True, True)),
                                               ((.025, .025), (.05, .05), (True, True)),
                                               ((None, .01), (None, .02), (None, True)),
                                               ((None, None), (None, None), (None, None))])
def test_holm_exact_two_test_family(raw, adjusted, reject):
    result = stats.holm_primary(dict(zip(stats.PRIMARY_FAMILY, raw)))
    assert tuple(r.adjusted_p_value for r in result) == adjusted
    assert tuple(r.reject_at_005 for r in result) == reject


@pytest.mark.parametrize("values", [{}, {stats.PRIMARY_FAMILY[0]: .01},
                                    dict(zip(stats.PRIMARY_FAMILY, (np.nan, .2))),
                                    dict(zip(stats.PRIMARY_FAMILY, (-.1, .2))),
                                    dict(zip(stats.PRIMARY_FAMILY, (True, .2)))])
def test_invalid_holm_inputs(values):
    with pytest.raises(ValueError):
        stats.holm_primary(values)


@pytest.mark.parametrize("label", [-1, 2, True, "FRI", 0.])
def test_invalid_class_labels(label):
    with pytest.raises(ValueError):
        record(0, label, 1)


def test_duplicate_image_ids_rejected_even_if_metrics_differ():
    records = (record(0, 0, 1), record(0, 0, 2, metric="y"))
    for function in (lambda: stats.summarize(records, "x"),
                     lambda: stats.bootstrap_paired(records, ("x",)),
                     lambda: stats.primary_sign_flip(records, stats.PRIMARY_FAMILY[0])):
        with pytest.raises(ValueError, match="Duplicate"):
            function()


@pytest.mark.parametrize("value", [np.zeros((150, 150)), [1, 2], True, "1"])
def test_arrays_pixels_and_invalid_scalars_rejected(value):
    with pytest.raises(ValueError):
        stats.observation(value)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_scalar_unavailable(value):
    item = stats.observation(value)
    assert item.value is None and item.reasons == ("nonfinite",)


def test_invalid_status_and_mutable_or_forged_record_rejected():
    with pytest.raises(ValueError):
        record(0, 0, stats.Observation(2., "unavailable", ("bad",)))
    with pytest.raises(ValueError):
        record(0, 0, stats.Observation(np.nan, "available"))
    with pytest.raises(ValueError):
        stats.summarize((stats.ImageRecord(0, 0, 0, [("x", stats.observation(1))]),), "x")
    with pytest.raises(ValueError):
        stats.summarize((stats.ImageRecord(0, 0, 0, (("x", 1.),)),), "x")
    with pytest.raises(ValueError):
        stats.summarize((stats.ImageRecord(0, 0, 0, (["x", stats.observation(1)],)),), "x")
    for endpoints in ((), ("x", "x")):
        with pytest.raises(ValueError):
            stats.bootstrap_paired((), endpoints)


def test_primary_analysis_no_mutation_or_file_access():
    records = primary_records()
    before = deepcopy(records)
    with patch("builtins.open", side_effect=AssertionError("no artifact reads/writes")), \
         patch.object(Path, "open", side_effect=AssertionError("no artifact reads/writes")):
        result = stats.analyze_primary(records)
    assert records == before
    assert len(result.tests) == len(result.holm) == len(result.bootstrap) == 2
    assert tuple(t.endpoint for t in result.tests) == stats.PRIMARY_FAMILY
    assert all(r.coverage.class_counts == (3, 3) for r in result.bootstrap)
    assert all(len(dict(r.metrics)[stats.RANDOM].repeats) == 20 for r in result.records)


def test_primary_analysis_all_missing_returns_no_invented_inference():
    records = (stats.image_record(0, 0, None, {}), stats.image_record(1, 1, None, {}))
    result = stats.analyze_primary(records)
    assert all(t.p_value is None for t in result.tests)
    assert all(h.adjusted_p_value is None and h.reject_at_005 is None for h in result.holm)
    assert all(s.overall.total == 2 and s.overall.available == 0 for s in result.summaries)


def test_bootstrap_exact_seed42_draws_keep_every_class_size():
    labels = np.array([0, 1, 0, 1, 1])
    rng = np.random.Generator(np.random.PCG64(42))
    expected = np.concatenate((rng.choice([0, 2], (10000, 2), replace=True),
                               rng.choice([1, 3, 4], (10000, 3), replace=True)), axis=1)
    np.testing.assert_array_equal(stats._bootstrap_indices(labels), expected)


def test_mutable_source_measurements_are_not_changed_or_retained():
    raw = {seed: float(seed) for seed in range(2000, 2010)}
    before = deepcopy(raw)
    aggregate = stats.aggregate_repeats(raw, "noise", .005)
    assert raw == before
    raw[2000] = -99
    assert aggregate.repeats[0][1].value == 2000
    metrics = {"x": aggregate}
    r = stats.image_record(0, 0, 0, metrics)
    metrics["x"] = None
    assert dict(r.metrics)["x"] == aggregate
