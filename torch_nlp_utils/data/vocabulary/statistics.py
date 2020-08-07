from typing import (
    Dict, List, Union,
    Any, Type, T
)
import json
from overrides import overrides
from collections import defaultdict
from torch_nlp_utils.common import Registrable


class Statistics(Registrable):
    """
    Calculate Statistics for the Namespace.
    Currently supports only token frequency.
    """
    def __init__(self) -> None:
        self._frequencies = defaultdict(int)

    def save(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(self.get_statistics()['frequency'], file, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls: Type[T], path: str) -> T:
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            statistics = cls()
            statistics._frequencies = defaultdict(int, data)
        return statistics

    def update_stats(self, tokens: Union[List[str], List[int]]) -> None:
        """Update statistics with `tokens`."""
        for token in tokens:
            self._frequencies[token] += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get all statistics as dictionary."""
        all_statistics = {}
        all_statistics['frequency'] = self._frequencies
        all_statistics['min'] = min(self._frequencies.items(), key=lambda x: x[1])
        all_statistics['max'] = max(self._frequencies.items(), key=lambda x: x[1])
        all_statistics['average'] = sum(self._frequencies.values()) / len(self._frequencies)
        return all_statistics


@Statistics.register('target')
class TargetStatistics(Statistics):
    """Statistics for target Namespace in case of OneClass and MultiClass target."""
    pass


@Statistics.register('multilabel_target')
class MultiLabelTargetStatistics(TargetStatistics):
    """Statistics for target Namespace in case of MultiLabel target."""
    @overrides
    def update_stats(self, tokens: Union[List[str], List[int]]) -> None:
        """Update statistics with `tokens`."""
        for idx, token in enumerate(tokens):
            self._frequencies[idx] += token


@Statistics.register('regression_target')
class RegressionTargetStatistics(TargetStatistics):
    """Statistics for target Namespace in case of Regression target."""
    def __init__(self) -> None:
        super().__init__()
        self._min = 1e32
        self._max = -1e32
        self._sum_over_values = 0
        self._number_of_values = 0

    @overrides
    def save(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(self.get_statistics(), file, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls: Type[T], path: str) -> T:
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            statistics = cls()
            statistics._min = data['min']
            statistics._max = data['max']
            statistics._sum_over_values = data['sum']
            statistics._number_of_values = data['n']
        return statistics

    @overrides
    def update_stats(self, tokens: List[int]) -> None:
        """Update statistics with `tokens`."""
        self._min = min(self._min, min(tokens))
        self._max = max(self._max, max(tokens))
        self._sum_over_values += sum(tokens)
        self._number_of_values += len(tokens)

    @overrides
    def get_statistics(self) -> Dict[str, Any]:
        """Get all statistics as dictionary."""
        all_statistics = {}
        all_statistics['min'] = self._min
        all_statistics['max'] = self._max
        all_statistics['average'] = self._sum_over_values / self._number_of_values
        all_statistics['sum'] = self._sum_over_values
        all_statistics['n'] = self._number_of_values
        return all_statistics
