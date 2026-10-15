"""Baseline public-input diagnostic copied into the first research parent."""


def verify(snapshot):
    assert 1 <= len(snapshot["images"]) <= 3
    assert all(image["frame"] <= snapshot["frame"] for image in snapshot["images"])
    return True
