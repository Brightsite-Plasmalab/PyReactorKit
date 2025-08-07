from pyreactorkit.models.base import PlugFlowReactor


class ExchangingReactor:
    def __init__(self, **kwargs):
        super(ExchangingReactor, self).__init__(**kwargs)
        self._exchange = None

    @property
    def exchange(self):
        return self._exchange

    @exchange.setter
    def exchange(self, value):
        self._exchange = value

    def _get_exchange(self):
        return self._exchange

    def _set_exchange(self, value):
        self._exchange = value

    exchange = property(_get_exchange, _set_exchange)
