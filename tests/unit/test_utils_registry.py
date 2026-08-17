import pytest

from nssc.utils.registry import Registry


def test_register_and_build():
    R = Registry("test")

    @R.register("a")
    class A:
        def __init__(self, x=1):
            self.x = x

    assert "a" in R and R.keys() == ["a"]
    assert R.build("a", x=3).x == 3
    assert A.registry_key == "a"


def test_duplicate_key_raises():
    R = Registry("t")

    @R.register("k")
    class A:  # noqa
        pass

    with pytest.raises(KeyError):

        @R.register("k")
        class B:  # noqa
            pass


def test_unknown_key():
    with pytest.raises(KeyError):
        Registry("t").get("nope")
