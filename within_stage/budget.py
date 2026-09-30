"""Fresh twelve-hour authorization, reusing the unchanged reservation engine."""
from pathlib import Path
import sqlite3
import time

from learning_lab.budget import ResearchBudget, _integer


class WithinStageBudget(ResearchBudget):
    @classmethod
    def create(cls, path, *, max_interactions=4000000, wall_seconds=43200, started_epoch=None):
        _integer(max_interactions, 'max_interactions', minimum=1)
        _integer(wall_seconds, 'wall_seconds', minimum=1, maximum=43200)
        now = time.time()
        started_epoch = now if started_epoch is None else started_epoch
        if (isinstance(started_epoch, bool) or not isinstance(started_epoch, (int, float))
                or not 0 <= started_epoch <= now):
            raise ValueError('started_epoch must be a finite past timestamp')
        path = Path(path).resolve()
        path.touch(exist_ok=False)
        connection = sqlite3.connect(path)
        try:
            connection.execute('PRAGMA journal_mode=WAL')
            connection.execute('PRAGMA synchronous=FULL')
            connection.executescript('''
                CREATE TABLE metadata (
                    version INTEGER NOT NULL, max_interactions INTEGER NOT NULL,
                    wall_seconds INTEGER NOT NULL, started_epoch REAL NOT NULL,
                    started_monotonic REAL NOT NULL
                );
                CREATE TABLE authorization (
                    kind TEXT NOT NULL, maximum_wall_seconds INTEGER NOT NULL,
                    maximum_interactions INTEGER NOT NULL
                );
                CREATE TABLE reservations (
                    id TEXT PRIMARY KEY, label TEXT NOT NULL UNIQUE,
                    reserved INTEGER NOT NULL CHECK(reserved>0),
                    actual INTEGER CHECK(actual>=0 AND actual<=reserved),
                    created_epoch REAL NOT NULL, settled_epoch REAL
                );
            ''')
            connection.execute('INSERT INTO metadata VALUES(1,?,?,?,?)', (
                max_interactions, wall_seconds, started_epoch,
                time.monotonic() - (now - started_epoch),
            ))
            connection.execute('INSERT INTO authorization VALUES(?,?,?)', (
                'new-within-stage-tranche-not-extension-of-closed-study', 43200, 4000000,
            ))
            connection.commit()
        finally:
            connection.close()
        return cls(path)
