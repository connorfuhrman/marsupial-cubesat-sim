"""Ray actor pool."""


import ray
import warnings
from os import getenv


class ActorPool:
    """Class which manages Ray actors.

    This class will launch some number of concurrant
    Ray actors, await the results of 1, then launch
    another until the desired number of Actors
    have run.
    """

    def __init__(
        self,
        make_actor: callable,
        run_actor: callable,
        n_experiments: int,
        n_concurrent: int = -1,
    ):
        """Initialize the actor pool.

        The pool needs some callable to make an actor and another
        to run that actor which must return a Ray object ref.
        Additionally, set the number of experiments to run and
        how many actors at one to run (if -1 default to number of
        CPU cores as reported by Ray).

        make_actor must accept no arguments.
        run_actor must accept 1 argument which is the actor to run.
        """
        if not ray.is_initialized():
            if not self.supress_init_warn:
                warnings.warn(
                    "Initializing ray without arguments. "
                    "If custom initialization is required then "
                    "initialize Ray before constructing this object. "
                    "Export ACTOR_POOL_NO_RAY_INIT_WARN=1 to supress"
                )
            ray.init()

        self._make_actor = make_actor
        self._run_actor = run_actor
        self._n_experiments = n_experiments
        if n_concurrent == -1:
            n_concurrent = int(ray.available_resources()["CPU"])
        self._n_concurrent = n_concurrent

        self._refs_to_actors = dict()
        self._results = []

    def run(self):
        """Run the requested number of experiments and return the results."""
        actors = [self._make_actor() for _ in range(self._n_concurrent)]
        for a in actors:
            ref = self._run_actor(a)
            self._refs_to_actors[ref] = a
        while self.is_running:
            dones, running = ray.wait(self.ray_refs)
            for d in dones:
                self._results.append(ray.get(d))
                del self._refs_to_actors[d]
            if len(self._results) + len(running) < self._n_experiments:
                actor = self._make_actor()
                ref = self._run_actor(actor)
                self._refs_to_actors[ref] = actor
        return self._results

    @property
    def is_running(self) -> bool:  # noqa D
        return len(self._results) != self._n_experiments

    @property
    def ray_refs(self):  # noqa D
        return list(self._refs_to_actors.keys())

    @property
    def supress_init_warn(self) -> bool:  # noqa D:
        if (v := getenv("ACTOR_POOL_NO_RAY_INIT_WARN")) is not None:
            return v == 1
        else:
            return False


if __name__ == "__main__":
    import time
    import numpy as np

    @ray.remote
    class TestActor:
        """An example of a Ray actor.

        This actor sleeps for between 2 and 6 seconds,
        prints out it's msg, then returns that msg.
        """

        def __init__(self, msg):  # noqa D
            self.msg = msg

        def do_something(self):  # noqa D
            print("Doing something!!")
            time.sleep(np.random.randint(2, 6))
            print(f"Done with my task. Returning {self.msg}")
            return self.msg

    class StatefulTestActorCreator:
        """An example of a stateful callable.

        If you need to have some state preserved between
        cration of actors, e.g., to chose from a set of
        parameters for the Actor's construction, you
        may define a class with a callable method accepting
        no arguments as such.
        """

        def __init__(self, msgs):  # noqa D
            self.msgs = msgs
            self.idx = 0

        def __call__(self):  # noqa D
            a = TestActor.remote(self.msgs[self.idx])
            self.idx += 1
            self.idx %= len(self.msgs)
            return a

    msgs = [
        "Hello world",
        "from the test",
        "of the ray actor pool!",
        "(these messages will not appear in order)",
    ]
    creator = StatefulTestActorCreator(msgs)

    pool = ActorPool(creator, lambda a: a.do_something.remote(), n_experiments=25)

    res = pool.run()
    print(f"Got results {res}")
