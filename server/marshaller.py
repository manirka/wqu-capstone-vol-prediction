import datetime
import logging.config
import json
import pandas as pd

log = logging.getLogger(__name__)


class Marshaller:
    def __init__(self):
        pass

    def marshall(self, message):
        pass


class Unmarshaller:
    def __init__(self):
        pass

    def unmarshall(self, request):
        pass


class MessageConverter(Marshaller, Unmarshaller):
    def __init__(self):
        super().__init__()
        self._unmarshallers = {}
        self._marshallers = {}

    def register_unmarshaller(self, action_name, unmarshaller):
        self._unmarshallers[action_name] = unmarshaller

    def register_marshaller(self, response_type, marshaller):
        self._marshallers[response_type] = marshaller

    def unmarshall(self, request):
        parsed = json.loads(request)
        unmarshaller = self._unmarshallers.get(parsed['action'])
        if not unmarshaller:
            raise ValueError(f"Unmarshaller not found for action {parsed['action']}")
        return unmarshaller.unmarshall(parsed)

    def marshall(self, message):
        marshaller = self._marshallers.get(message.__class__.__name__)
        if not marshaller:
            raise ValueError(f'Marshaller not found for class {message.__class__.__name__}')
        return marshaller.marshall(message)


class BarUnmarshaller(Unmarshaller):
    def unmarshall(self, request):
        return pd.Series({f: request[f] for f in ['ticker', 'open', 'close', 'volume']},
                      name=datetime.datetime.strptime(request['datetime'], '%Y-%m-%d %H:%M:%S'))

class SubUnmarshaller(Unmarshaller):
    def unmarshall(self, request):
        return pd.Series({f: request[f] for f in ['ticker', 'start', 'end']}, name='subscription')

class NumberMarshaller(Marshaller):
    def marshall(self, value):
        return {'type': 'result', 'value': value}
