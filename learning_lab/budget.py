"""Durable interaction reservations; unfinished workers remain fully charged."""
from contextlib import contextmanager
from pathlib import Path
import sqlite3
import time
import uuid


class BudgetExhausted(RuntimeError):
    pass


def _integer(value, name, *, minimum=0, maximum=4000000):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f'{name} must be an integer in {minimum}..{maximum}')
    return value


class ResearchBudget:
    def __init__(self, path):
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        with self._transaction() as connection:
            self._metadata(connection)

    @classmethod
    def create(cls, path, *, max_interactions, wall_seconds, started_epoch=None):
        _integer(max_interactions, 'max_interactions', minimum=1)
        _integer(wall_seconds, 'wall_seconds', minimum=1, maximum=28800)
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
            connection.commit()
        finally:
            connection.close()
        return cls(path)

    @contextmanager
    def _transaction(self):
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute('PRAGMA synchronous=FULL')
            connection.execute('BEGIN IMMEDIATE')
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _metadata(connection):
        rows = connection.execute('SELECT * FROM metadata').fetchall()
        if len(rows) != 1 or rows[0]['version'] != 1:
            raise ValueError('Invalid research budget metadata')
        return rows[0]

    @staticmethod
    def _remaining_time(metadata):
        elapsed = time.monotonic() - metadata['started_monotonic']
        if elapsed < 0:
            return 0.0
        return max(0.0, min(
            metadata['started_epoch'] + metadata['wall_seconds'] - time.time(),
            metadata['wall_seconds'] - elapsed,
        ))

    def snapshot(self):
        with self._transaction() as connection:
            metadata = self._metadata(connection)
            rows = [dict(row) for row in connection.execute('SELECT * FROM reservations ORDER BY created_epoch')]
            charged = sum(row['reserved'] if row['actual'] is None else row['actual'] for row in rows)
            settled = sum(row['actual'] for row in rows if row['actual'] is not None)
            return {
                'max_interactions': metadata['max_interactions'], 'charged_interactions': charged,
                'settled_interactions': settled,
                'remaining_interactions': metadata['max_interactions'] - charged,
                'remaining_seconds': self._remaining_time(metadata),
                'deadline_epoch': metadata['started_epoch'] + metadata['wall_seconds'],
                'reservations': rows,
            }

    def reserve(self, label, interactions):
        _integer(interactions, 'interactions', minimum=1)
        if not isinstance(label, str) or not label.strip() or len(label) > 160:
            raise ValueError('label must be a nonempty string of at most160characters')
        with self._transaction() as connection:
            metadata = self._metadata(connection)
            if self._remaining_time(metadata) <= 0:
                raise BudgetExhausted('Research wall-time budget exhausted')
            charged = connection.execute(
                'SELECT COALESCE(SUM(COALESCE(actual,reserved)),0) FROM reservations').fetchone()[0]
            if charged + interactions > metadata['max_interactions']:
                raise BudgetExhausted('Research interaction budget exhausted')
            token = uuid.uuid4().hex
            connection.execute(
                'INSERT INTO reservations(id,label,reserved,created_epoch) VALUES(?,?,?,?)',
                (token, label, interactions, time.time()))
        return token

    def settle(self, token, interactions):
        _integer(interactions, 'actual interactions')
        with self._transaction() as connection:
            row = connection.execute('SELECT * FROM reservations WHERE id=?', (token,)).fetchone()
            if row is None or interactions > row['reserved']:
                raise ValueError('Unknown reservation or actual interactions exceed reservation')
            if row['actual'] is not None:
                if row['actual'] != interactions:
                    raise ValueError('Conflicting settlement of an already settled reservation')
                return
            connection.execute(
                'UPDATE reservations SET actual=?,settled_epoch=? WHERE id=?',
                (interactions, time.time(), token))
