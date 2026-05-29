import config  # noqa: F401 - adds cool-conditions/python to sys.path
from intervals import interval_entry
from time_utils import format_time


class LHCStableReader:
    def __init__(self, db_file, verbose=False):
        self.db_file = db_file
        self.verbose = verbose
        self.backend = "cool"
        self.db = None
        self.folder = None
        self.channel = None
        self.sqlite = None

        try:
            import CoolConvUtilities.AtlCoolLib as AtlCoolLib
            from PyCool import cool
        except ImportError:
            self._open_sqlite()
            return

        db_name = f"sqlite://;schema={db_file};dbname=CONDBR3"
        self.db = AtlCoolLib.readOpen(db_name)
        if not self.db:
            raise RuntimeError(f"Error opening {db_file}")
        self.folder = self.db.getFolder("/LHC/BeamData")
        if not self.folder:
            self.db.closeDatabase()
            raise RuntimeError("Error opening /LHC/BeamData")
        self.channel = cool.ChannelSelection.all()

    def _open_sqlite(self):
        import sqlite3

        self.backend = "sqlite"
        self.sqlite = sqlite3.connect(self.db_file)
        cursor = self.sqlite.cursor()
        cursor.execute(
            "select FOLDER_IOVTABLENAME from CONDBR3_NODES "
            "where NODE_FULLPATH = '/LHC/BeamData'"
        )
        row = cursor.fetchone()
        if row is None:
            self.sqlite.close()
            raise RuntimeError("Error opening /LHC/BeamData in sqlite DB")
        self.iov_table = row[0]
        if self.verbose:
            print(f"Using sqlite fallback for /LHC/BeamData: {self.iov_table}")

    def close(self):
        if self.backend == "cool":
            self.db.closeDatabase()
        elif self.sqlite is not None:
            self.sqlite.close()

    def stable_intervals(self, since, until):
        if self.backend == "sqlite":
            return self._stable_intervals_sqlite(since, until)

        iov_since = 1000 * int(since * 1e6)
        iov_until = 1000 * int(until * 1e6)
        stable_list = []
        iterator = self.folder.browseObjects(iov_since, iov_until, self.channel)
        try:
            while iterator.goToNext():
                obj = iterator.currentRef()
                payload = obj.payload()
                beam_mode = payload["BeamMode"]
                if self.verbose:
                    print(
                        f"LHC fill {payload['FillNumber']} mode {beam_mode} "
                        f"until {obj.until() / 1e9}"
                    )
                if beam_mode != "STABLE BEAMS":
                    continue

                start = max(obj.since() / 1e9, since)
                stop = min(obj.until() / 1e9, until)
                if start >= stop:
                    continue

                fill = payload["FillNumber"]
                if stable_list:
                    previous = stable_list[-1]
                    if previous["lhc_fill"] == fill and previous["stop_utime"] == start:
                        previous["stop_utime"] = stop
                        previous["stop_time"] = format_time(stop)
                        continue

                entry = interval_entry(start, stop)
                entry["lhc_fill"] = fill
                stable_list.append(entry)
        finally:
            iterator.close()
        return stable_list

    def _stable_intervals_sqlite(self, since, until):
        iov_since = int(since * 1e9)
        iov_until = int(until * 1e9)
        stable_list = []
        cursor = self.sqlite.cursor()
        cursor.execute(
            f"select IOV_SINCE, IOV_UNTIL, FillNumber, BeamMode from {self.iov_table} "
            "where IOV_UNTIL > ? and IOV_SINCE < ? order by IOV_SINCE",
            (iov_since, iov_until),
        )
        for row in cursor.fetchall():
            obj_since, obj_until, fill, beam_mode = row
            if self.verbose:
                print(f"LHC fill {fill} mode {beam_mode} until {obj_until / 1e9}")
            if beam_mode != "STABLE BEAMS":
                continue

            start = max(obj_since / 1e9, since)
            stop = min(obj_until / 1e9, until)
            if start >= stop:
                continue

            if stable_list:
                previous = stable_list[-1]
                if previous["lhc_fill"] == fill and previous["stop_utime"] == start:
                    previous["stop_utime"] = stop
                    previous["stop_time"] = format_time(stop)
                    continue

            entry = interval_entry(start, stop)
            entry["lhc_fill"] = fill
            stable_list.append(entry)
        return stable_list
