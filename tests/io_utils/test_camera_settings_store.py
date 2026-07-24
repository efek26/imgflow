from imgflow.io_utils.camera_settings_store import (
    list_settings,
    load_last_used,
    load_settings,
    save_last_used,
    save_settings,
)


def test_save_and_load_settings_round_trip(tmp_path):
    values = {"ExposureTime": 5000.0, "Gain": 2.5, "TriggerMode": "On", "ReverseX": True}
    save_settings("hat1", values, directory=tmp_path)

    data = load_settings("hat1", directory=tmp_path)

    assert data.values == values
    assert data.created_at is not None
    assert data.operator_note is None


def test_list_settings_returns_sorted_names(tmp_path):
    save_settings("b_profile", {}, directory=tmp_path)
    save_settings("a_profile", {}, directory=tmp_path)

    assert list_settings(tmp_path) == ["a_profile", "b_profile"]


def test_list_settings_empty_when_directory_missing(tmp_path):
    assert list_settings(tmp_path / "does_not_exist") == []


def test_save_defaults_created_at_when_not_given(tmp_path):
    save_settings("hat2", {"Gain": 1.0}, directory=tmp_path)

    data = load_settings("hat2", directory=tmp_path)

    assert data.created_at is not None
    assert data.operator_note is None


def test_save_accepts_explicit_created_at_and_operator_note(tmp_path):
    save_settings(
        "hat3",
        {"Gain": 1.0},
        directory=tmp_path,
        created_at="2026-01-15T10:00:00",
        operator_note="Efe - ilk kurulum",
    )

    data = load_settings("hat3", directory=tmp_path)

    assert data.created_at == "2026-01-15T10:00:00"
    assert data.operator_note == "Efe - ilk kurulum"


def test_resaving_overwrites_previous_values(tmp_path):
    save_settings("hat4", {"Gain": 1.0}, directory=tmp_path)
    save_settings("hat4", {"Gain": 2.0}, directory=tmp_path)

    data = load_settings("hat4", directory=tmp_path)
    assert data.values == {"Gain": 2.0}


def test_save_and_load_last_used_round_trip(tmp_path):
    save_last_used({"Gain": 3.5}, directory=tmp_path)

    data = load_last_used(tmp_path)

    assert data is not None
    assert data.values == {"Gain": 3.5}


def test_load_last_used_returns_none_when_missing(tmp_path):
    assert load_last_used(tmp_path) is None


def test_last_used_hidden_from_list_settings(tmp_path):
    save_last_used({"Gain": 3.5}, directory=tmp_path)
    save_settings("hat5", {"Gain": 1.0}, directory=tmp_path)

    assert list_settings(tmp_path) == ["hat5"]
