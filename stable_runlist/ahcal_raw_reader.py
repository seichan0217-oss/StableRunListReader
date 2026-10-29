"""Minimal standalone reader for AHCAL DAQ raw trigger records.

This module intentionally uses only the Python standard library.  The binary
layouts mirror EventFormats/DAQFormats.hpp and FAIR's BinaryRawHitReader, but
no FAIR, ROOT, or PyROOT library is imported at runtime.
"""

from dataclasses import dataclass
from pathlib import Path
import struct
import zlib


EVENT_MARKER = 0xBB
EVENT_VERSION = 0x01
EVENT_HEADER = struct.Struct("<BBHBBHIB3sQQHHQ")
EVENT_HEADER_SIZE = EVENT_HEADER.size

FRAGMENT_MARKER = 0xAA
FRAGMENT_VERSION = 0x0001
FRAGMENT_HEADER = struct.Struct("<BBHHHIIQHHQ")
FRAGMENT_HEADER_SIZE = FRAGMENT_HEADER.size

TLU_SOURCE_ID = 0x070000
TLU_PAYLOAD_SIZE = 20
EVENT_STATUS_COMPRESSED = 1 << 11

COMPRESSION_NONE = 0x00
COMPRESSION_ZSTD = 0x01
COMPRESSION_ZLIB = 0x02
COMPRESSION_LZ4 = 0x03


class RawFormatError(RuntimeError):
    """Raised when a raw event does not match the supported DAQ format."""


class UnsupportedCompressionError(RawFormatError):
    """Raised for compression algorithms unavailable in the standard library."""


@dataclass(frozen=True)
class RawTriggerEvent:
    path: str
    file_event_index: int
    run_number: int
    event_id: int
    event_counter: int
    cpu_timestamp_us: int
    trigger_id: int
    tlu_timestamp: int
    inputs: tuple
    fine_timestamps: tuple

    @property
    def unix_time(self):
        return self.cpu_timestamp_us / 1_000_000.0


@dataclass
class RawReadStats:
    events_seen: int = 0
    events_with_tlu: int = 0
    events_without_tlu: int = 0
    malformed_events: int = 0


def _read_exact(stream, size, description, allow_eof=False):
    data = stream.read(size)
    if allow_eof and not data:
        return None
    if len(data) != size:
        raise RawFormatError(
            f"truncated {description}: expected {size} bytes, got {len(data)}"
        )
    return data


def _decode_run_number(raw_three_bytes):
    return int.from_bytes(raw_three_bytes, byteorder="little", signed=False)


def _decompress_payload(payload, compression_code):
    if compression_code == COMPRESSION_ZLIB:
        try:
            return zlib.decompress(payload)
        except zlib.error as error:
            raise RawFormatError(f"zlib decompression failed: {error}") from error
    names = {
        COMPRESSION_ZSTD: "zstd",
        COMPRESSION_LZ4: "lz4",
    }
    name = names.get(compression_code, f"code {compression_code}")
    raise UnsupportedCompressionError(
        f"{name} compressed raw events require a non-standard decompressor; "
        "use an uncompressed raw file or convert it outside the Athena process"
    )


def _decode_tlu_payload(payload):
    if len(payload) != TLU_PAYLOAD_SIZE:
        raise RawFormatError(
            f"unexpected TLU payload size {len(payload)}; "
            f"expected {TLU_PAYLOAD_SIZE}"
        )
    word0, word1, word2, word3, word4 = struct.unpack("<5I", payload)
    timestamp = ((word0 & 0xFFFF) << 32) | word1
    inputs = tuple((word0 >> bit) & 1 for bit in range(16, 22))
    fine_timestamps = (
        (word2 >> 24) & 0xFF,
        (word2 >> 16) & 0xFF,
        (word2 >> 8) & 0xFF,
        word2 & 0xFF,
        (word4 >> 24) & 0xFF,
        (word4 >> 16) & 0xFF,
    )
    return word3, timestamp, inputs, fine_timestamps


class RawFileReader:
    """Stream trigger records from one DAQ raw file."""

    def __init__(self, path, max_payload_bytes=1_000_000):
        self.path = Path(path)
        self.max_payload_bytes = max_payload_bytes
        self.stats = RawReadStats()

    def __iter__(self):
        with self.path.open("rb") as stream:
            event_index = 0
            while True:
                try:
                    event = self._read_event(stream, event_index)
                except (RawFormatError, UnsupportedCompressionError):
                    self.stats.malformed_events += 1
                    raise
                if event is None:
                    break
                self.stats.events_seen += 1
                if event is False:
                    self.stats.events_without_tlu += 1
                else:
                    self.stats.events_with_tlu += 1
                    yield event
                event_index += 1

    def _read_event(self, stream, event_index):
        raw_header = _read_exact(
            stream,
            EVENT_HEADER_SIZE,
            f"event header in {self.path}",
            allow_eof=True,
        )
        if raw_header is None:
            return None

        (
            marker,
            _event_tag,
            _trigger_bits,
            version,
            compression_code,
            header_size,
            payload_size,
            fragment_count,
            run_number_bytes,
            event_id,
            event_counter,
            _bc_id,
            status,
            cpu_timestamp_us,
        ) = EVENT_HEADER.unpack(raw_header)

        if marker != EVENT_MARKER:
            raise RawFormatError(
                f"{self.path}: event {event_index}: marker 0x{marker:02x}, "
                f"expected 0x{EVENT_MARKER:02x}"
            )
        if version != EVENT_VERSION:
            raise RawFormatError(
                f"{self.path}: event {event_index}: unsupported event version {version}"
            )
        if header_size < EVENT_HEADER_SIZE:
            raise RawFormatError(
                f"{self.path}: event {event_index}: invalid header size {header_size}"
            )
        if payload_size > self.max_payload_bytes:
            raise RawFormatError(
                f"{self.path}: event {event_index}: payload size {payload_size} "
                f"exceeds {self.max_payload_bytes}"
            )

        if header_size > EVENT_HEADER_SIZE:
            _read_exact(
                stream,
                header_size - EVENT_HEADER_SIZE,
                f"extended event header in {self.path}",
            )
        payload = _read_exact(stream, payload_size, f"event payload in {self.path}")
        if status & EVENT_STATUS_COMPRESSED:
            payload = _decompress_payload(payload, compression_code)
        elif compression_code != COMPRESSION_NONE:
            raise RawFormatError(
                f"{self.path}: event {event_index}: compression code "
                f"{compression_code} without compressed status"
            )

        offset = 0
        tlu_values = None
        for fragment_index in range(fragment_count):
            if len(payload) - offset < FRAGMENT_HEADER_SIZE:
                raise RawFormatError(
                    f"{self.path}: event {event_index}: truncated fragment "
                    f"header {fragment_index}"
                )
            fragment_header = FRAGMENT_HEADER.unpack_from(payload, offset)
            (
                fragment_marker,
                _fragment_tag,
                _fragment_trigger_bits,
                fragment_version,
                fragment_header_size,
                fragment_payload_size,
                source_id,
                _fragment_event_id,
                _fragment_bc_id,
                _fragment_status,
                _fragment_timestamp,
            ) = fragment_header
            if fragment_marker != FRAGMENT_MARKER:
                raise RawFormatError(
                    f"{self.path}: event {event_index}: fragment {fragment_index} "
                    f"marker 0x{fragment_marker:02x}, expected "
                    f"0x{FRAGMENT_MARKER:02x}"
                )
            if fragment_version != FRAGMENT_VERSION:
                raise RawFormatError(
                    f"{self.path}: event {event_index}: unsupported fragment "
                    f"version {fragment_version}"
                )
            if fragment_header_size < FRAGMENT_HEADER_SIZE:
                raise RawFormatError(
                    f"{self.path}: event {event_index}: invalid fragment header "
                    f"size {fragment_header_size}"
                )
            fragment_end = (
                offset + fragment_header_size + fragment_payload_size
            )
            if fragment_end > len(payload):
                raise RawFormatError(
                    f"{self.path}: event {event_index}: fragment {fragment_index} "
                    "extends beyond the event payload"
                )
            if source_id == TLU_SOURCE_ID:
                fragment_payload = payload[
                    offset + fragment_header_size : fragment_end
                ]
                tlu_values = _decode_tlu_payload(fragment_payload)
            offset = fragment_end

        if offset != len(payload):
            raise RawFormatError(
                f"{self.path}: event {event_index}: decoded {offset} of "
                f"{len(payload)} payload bytes"
            )
        if tlu_values is None:
            return False

        trigger_id, tlu_timestamp, inputs, fine_timestamps = tlu_values
        return RawTriggerEvent(
            path=str(self.path),
            file_event_index=event_index,
            run_number=_decode_run_number(run_number_bytes),
            event_id=event_id,
            event_counter=event_counter,
            cpu_timestamp_us=cpu_timestamp_us,
            trigger_id=trigger_id,
            tlu_timestamp=tlu_timestamp,
            inputs=inputs,
            fine_timestamps=fine_timestamps,
        )
