from .communication import SimulationManager as CommsSimManager, Transmission

from .messages import Message, SpacecraftState as SpacecraftStateMsg

# Check that there are no conflicting messages
#
# This needs to be done to properly do any message callback dispatch.
# Messages are defined as TypedDicts and therefore to match if a dict is
# if a certain msg type at runtime we check that dict's keys vs the type's
# annotation keys. Here we check that there are no messages with duplicate
# keys which would inject a nasty bug into the system.
import inspect
import itertools
import importlib
import deepdiff
import DSRC.simulation.communication.messages

# _msgs_mod = 'DSRC.simulation.communication.messages'
_msgs_mod = importlib.import_module("DSRC.simulation.communication.messages")
_classes = [cls_name for cls_name, cls_obj in inspect.getmembers(_msgs_mod) if inspect.isclass(cls_obj)]
_classes = filter(lambda c: c != 'Message' and c != 'MessageData', _classes)

MessageData_cls = getattr(_msgs_mod, 'MessageData')
MessageData_keys = list(MessageData_cls.__annotations__.keys())


def _class_same(c1, c2, attr):
    def get(c, a):
        a = getattr(c.__annotations__, a)
        c_mems = a()
        n_cmems = len(c_mems)
        return set(c_mems), n_cmems

    def check(c, n, cls):
        if len(c) != n:
            raise ImportError("Somehow got repeated TypedDict keys")
        for m in MessageData_keys:
            if m not in c:
                raise ImportError(f"{m} is not found in the keys of {cls}. "
                                  "Does this type inherit from MessageData as it should?")
    c1_mems, nc1_mems = get(c1, attr)
    c2_mems, nc2_mems = get(c2, attr)

    check(c1_mems, nc1_mems, c1)
    check(c2_mems, nc2_mems, c2)

    return c1_mems == c2_mems


for c1, c2 in itertools.combinations(_classes, 2):
    cls1 = getattr(_msgs_mod, c1)
    cls2 = getattr(_msgs_mod, c2)

    def mem_names_same():  # noqa D
        return _class_same(cls1, cls2, 'keys')

    if mem_names_same():
        raise ImportError(f"Classes {c1} and {c2} have identical keys: {list(cls1.__annotations__.keys())}. "
                          "This is not consistent with the requirements since "
                          "the DSRC communication simulation backend checks if "
                          "a dictionary has the same keys as a TypedDict to "
                          "determine if it's of a given type. "
                          "You need to update the message type so that it's keys "
                          "are not identical to another message type.")
