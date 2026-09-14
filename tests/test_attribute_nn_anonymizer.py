import pytest

from identity_anonymizer.anonymizers import AttributeNNAnonymizer, get_anonymizer, list_anonymizers


def test_attribute_nn_is_registered_by_default():
    assert "attribute_nn" in list_anonymizers()
    assert isinstance(get_anonymizer("attribute_nn"), AttributeNNAnonymizer)


def test_anonymize_single_attribute_dict():
    model = AttributeNNAnonymizer(hidden_dim=8, seed=0)
    output = model.anonymize({"age": 25, "gender": 0, "race": 2})
    assert output.shape == (1, model.output_dim)


def test_anonymize_batch_of_attribute_dicts():
    model = AttributeNNAnonymizer(hidden_dim=8, seed=0)
    batch = [
        {"age": 10, "gender": 1, "race": 0},
        {"age": 40, "gender": 0, "race": 3},
        {"age": 70, "gender": 1, "race": 4},
    ]
    output = model.anonymize(batch)
    assert output.shape == (3, model.output_dim)


@pytest.mark.parametrize(
    "bad_input",
    [
        {"gender": 0, "race": 1},  # "age" が無い
        {"age": 25, "race": 1},  # "gender" が無い
        {"age": 25, "gender": 0},  # "race" が無い
        {"age": 25, "gender": 2, "race": 1},  # genderが不正な値
        {"age": 25, "gender": 0, "race": 5},  # raceが範囲外
    ],
)
def test_validate_input_rejects_invalid_attribute_values(bad_input):
    model = AttributeNNAnonymizer(hidden_dim=8, seed=0)
    with pytest.raises(ValueError):
        model.anonymize(bad_input)


def test_validate_input_rejects_non_dict_input():
    model = AttributeNNAnonymizer(hidden_dim=8, seed=0)
    with pytest.raises(TypeError):
        model.anonymize("not a dict")


def test_validate_input_rejects_empty_batch():
    model = AttributeNNAnonymizer(hidden_dim=8, seed=0)
    with pytest.raises(ValueError):
        model.anonymize([])
