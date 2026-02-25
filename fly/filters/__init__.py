from functools import partial
from typing import List

from fly.api.filter import FilterEnsemble
from fly.api.registry import get_filter

from . import custom, extraction, selection, transformation


def build_filter_ensemble(
    filter_name: str, components: List[List[str]]
) -> FilterEnsemble:
    """
    Create a filtering pipeline.
    """
    filters = []
    for function, kwargs in components:
        if kwargs is None:
            kwargs = {}
        f = partial(get_filter(function), **kwargs)
        filters.append(f)

    return FilterEnsemble(name=filter_name, filters=filters)
