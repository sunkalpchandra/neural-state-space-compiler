from nssc.utils.config import Config, deep_merge, load_config, parse_override, save_yaml


def test_deep_merge_nested():
    a = {"x": {"y": 1, "z": 2}, "k": 1}
    b = {"x": {"y": 5}, "n": 3}
    m = deep_merge(a, b)
    assert m == {"x": {"y": 5, "z": 2}, "k": 1, "n": 3}
    assert a["x"]["y"] == 1  # no mutation


def test_base_inheritance_and_override(tmp_path):
    save_yaml({"a": 1, "b": {"c": 2}}, tmp_path / "base.yaml")
    save_yaml({"_base_": "base.yaml", "b": {"d": 3}}, tmp_path / "child.yaml")
    cfg = load_config(tmp_path / "child.yaml", overrides=["b.c=10", "e.f=[1,2]"])
    assert cfg.a == 1 and cfg.b.c == 10 and cfg.b.d == 3 and cfg.e.f == [1, 2]
    assert isinstance(cfg.b, Config)


def test_hash_stable_and_sensitive():
    c1 = Config({"a": 1, "b": {"c": 2}})
    c2 = Config({"b": {"c": 2}, "a": 1})
    c3 = Config({"a": 1, "b": {"c": 3}})
    assert c1.hash() == c2.hash() != c3.hash()


def test_parse_override_types():
    assert parse_override("x=1.5") == ("x", 1.5)
    assert parse_override("x=true") == ("x", True)
    assert parse_override("x=abc") == ("x", "abc")
