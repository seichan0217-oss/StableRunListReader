import config  # noqa: F401 - adds cool-conditions/python to sys.path
from intervals import interval_entry, merge_intervals


class LuminosityGapReader:
    def __init__(self, tag, acct_tag="", verbose=False):
        try:
            from CoolDataReader import CoolDataReader
        except ImportError as error:
            raise RuntimeError(
                "ATLAS luminosity gap checks require PyCool/CoolDataReader. "
                "Run in an ATLAS/Athena environment, or rerun with --no-lumi "
                "to build good intervals without ATLAS luminosity exclusions."
            ) from error

        self.tag = tag
        self.acct_tag = acct_tag
        self.verbose = verbose
        self.time_folder = None
        if self.acct_tag:
            self.lumi_folder = CoolDataReader(
                "COOLOFL_TRIGGER/CONDBR2",
                "/TRIGGER/OFLLUMI/LumiAccounting",
            )
        else:
            self.time_folder = CoolDataReader(
                "COOLONL_TRIGGER/CONDBR2",
                "/TRIGGER/LUMI/LBTIME",
            )
            self.lumi_folder = CoolDataReader(
                "COOLOFL_TRIGGER/CONDBR2",
                "/TRIGGER/OFLLUMI/OflPrefLumi",
            )

    def close(self):
        if self.time_folder is not None:
            self.time_folder.close()
        self.lumi_folder.close()

    def excluded_intervals(
        self,
        stable_list,
    ):
        excluded = []
        for stable in stable_list:
            excluded.extend(
                self._missing_luminosity_intervals(
                    stable["start_utime"],
                    stable["stop_utime"],
                )
            )
        return merge_intervals(excluded)

    def _missing_luminosity_intervals(
        self,
        since,
        until,
    ):
        if self.acct_tag:
            return self._missing_accounting_luminosity_intervals(since, until)

        iov_since = int(since * 1e9)
        iov_until = int(until * 1e9)

        self.time_folder.setIOVRange(iov_since, iov_until)
        self.time_folder.readData()

        time_by_runlb = {}
        for obj in self.time_folder.data:
            run = obj.payload()["Run"]
            lb = obj.payload()["LumiBlock"]
            runlb = (run << 32) | lb
            time_by_runlb[runlb] = (obj.since(), obj.until())

        runlb_list = sorted(time_by_runlb)
        if not runlb_list:
            return [interval_entry(since, until, "Missing ATLAS LBTIME")]

        self.lumi_folder.setIOVRange(runlb_list[0], runlb_list[-1])
        self.lumi_folder.setTag(self.tag)
        self.lumi_folder.setChannelId(0)
        self.lumi_folder.readData()

        if self.verbose:
            print(
                f"Read {len(self.time_folder.data)} LBTIME records and "
                f"{len(self.lumi_folder.data)} luminosity records"
            )

        excluded = []
        last_time = since
        found_lumi = False

        for obj in self.lumi_folder.data:
            runlb = obj.since()
            time_iov = time_by_runlb.get(runlb)
            if time_iov is None:
                if self.verbose:
                    run = runlb >> 32
                    lb = runlb & 0xFFFFFFFF
                    print(f"Skipping luminosity with no LBTIME for {run}/{lb}")
                continue

            block_since = max(time_iov[0] / 1e9, since)
            block_until = min(time_iov[1] / 1e9, until)
            if block_since >= block_until:
                continue

            found_lumi = True
            if block_since > last_time:
                excluded.append(
                    interval_entry(last_time, block_since, "Missing ATLAS luminosity")
                )
            last_time = max(last_time, block_until)

        if not found_lumi:
            return [interval_entry(since, until, "Missing ATLAS luminosity")]

        if last_time < until:
            excluded.append(interval_entry(last_time, until, "Missing ATLAS luminosity"))

        return excluded

    def _missing_accounting_luminosity_intervals(self, since, until):
        iov_since = int(since * 1e9)
        iov_until = int(until * 1e9)

        self.lumi_folder.setIOVRange(iov_since, iov_until)
        self.lumi_folder.setTag(self.acct_tag)
        self.lumi_folder.readData()

        if self.verbose:
            print(f"Read {len(self.lumi_folder.data)} accounting luminosity records")

        excluded = []
        last_time = since
        found_lumi = False

        for obj in self.lumi_folder.data:
            block_since = max(obj.since() / 1e9, since)
            block_until = min(obj.until() / 1e9, until)
            if block_since >= block_until:
                continue

            found_lumi = True
            if block_since > last_time:
                excluded.append(
                    interval_entry(last_time, block_since, "Missing ATLAS luminosity")
                )
            last_time = max(last_time, block_until)

        if not found_lumi:
            return [interval_entry(since, until, "Missing ATLAS luminosity")]

        if last_time < until:
            excluded.append(interval_entry(last_time, until, "Missing ATLAS luminosity"))

        return excluded
