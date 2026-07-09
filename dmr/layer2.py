"""Native DMR layer-2 metadata parsers."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from bitarray import bitarray
from bitarray.util import ba2int

from dmr.fec import qr_16_7_6_check


class LCSS(IntEnum):
    SingleFragmentLCorCSBK = 0
    FirstFragmentLC = 1
    LastFragmentLCorCSBK = 2
    ContinuationFragmentLCorCSBK = 3


@dataclass(frozen=True)
class EmbeddedSignalling:
    colour_code: int
    preemption_and_power_control_indicator: int
    link_control_start_stop: LCSS
    emb_parity: int
    emb_parity_ok: bool

    def to_extra(self) -> dict:
        return {
            "colour_code": self.colour_code,
            "preemption_power_indicator": self.preemption_and_power_control_indicator,
            "lcss": int(self.link_control_start_stop.value),
            "lcss_name": self.link_control_start_stop.name,
            "emb_parity": self.emb_parity,
            "emb_parity_ok": self.emb_parity_ok,
        }


@dataclass(frozen=True)
class ServiceOptions:
    value: int
    emergency: bool
    privacy: bool
    reserved: int
    broadcast: bool
    open_voice_call_mode: bool
    priority: int

    def to_extra(self) -> dict:
        return {
            "value": self.value,
            "emergency": self.emergency,
            "privacy": self.privacy,
            "reserved": self.reserved,
            "broadcast": self.broadcast,
            "open_voice_call_mode": self.open_voice_call_mode,
            "priority": self.priority,
        }


@dataclass(frozen=True)
class FullLinkControl:
    protect_flag: bool
    reserved_bit: int
    flco_value: int
    flco_name: str
    fid_value: int
    fid_name: str
    crc: int
    call_type: str = "unknown"
    service_options_value: int | None = None
    service_options: ServiceOptions | None = None
    source_address: int = 0
    group_address: int = 0
    target_address: int = 0
    position_error_value: int | None = None
    position_error_name: str = ""
    longitude: float | None = None
    latitude: float | None = None
    talker_alias_format_value: int | None = None
    talker_alias_format_name: str = ""
    talker_alias_data_length: int | None = None
    talker_alias_data_msb: bool | None = None
    talker_alias_data_hex: str = ""
    talker_alias_data_bits: str = ""

    def to_extra(self) -> dict:
        out = {
            "protect_flag": self.protect_flag,
            "reserved_bit": self.reserved_bit,
            "flco_value": self.flco_value,
            "flco_name": self.flco_name,
            "fid_value": self.fid_value,
            "fid_name": self.fid_name,
            "crc_value": self.crc,
            "call_type": self.call_type,
            "source_address": self.source_address,
            "group_address": self.group_address,
            "target_address": self.target_address,
        }
        if self.service_options is not None:
            out["service_options_value"] = self.service_options_value
            out["service_options"] = self.service_options.to_extra()
        if self.position_error_value is not None:
            out.update({
                "position_error_value": self.position_error_value,
                "position_error_name": self.position_error_name,
                "longitude": self.longitude,
                "latitude": self.latitude,
            })
        if self.talker_alias_format_value is not None:
            out.update({
                "talker_alias_format_value": self.talker_alias_format_value,
                "talker_alias_format_name": self.talker_alias_format_name,
                "talker_alias_data_length": self.talker_alias_data_length,
                "talker_alias_data_msb": self.talker_alias_data_msb,
                "talker_alias_data_hex": self.talker_alias_data_hex,
                "talker_alias_data_bits": self.talker_alias_data_bits,
            })
        return out


@dataclass(frozen=True)
class CSBK:
    last_block: bool
    protect_flag: bool
    csbko_value: int
    csbko_name: str
    fid_value: int
    fid_name: str
    crc: int
    source_address: int = 0
    target_address: int = 0
    bs_address: int = 0
    service_options_value: int | None = None
    service_options: ServiceOptions | None = None
    answer_response_value: int | None = None
    answer_response_name: str = ""
    additional_information_field: int | None = None
    source_type: int | None = None
    service_type_value: int | None = None
    service_type_name: str = ""
    reason_code: int | None = None
    csbk_content_follows_preambles: bool | None = None
    target_address_is_individual: bool | None = None
    blocks_to_follow: int | None = None
    sync_age: int | None = None
    generation: int | None = None
    leader_identifier: int | None = None
    new_leader: bool | None = None
    leader_dynamic_identifier: int | None = None
    channel_timing_opcode: int | None = None
    source_identifier: int | None = None
    source_dynamic_identifier: int | None = None
    tsccas_support: bool | None = None
    site_timeslot_synchronized: bool | None = None
    document_version_control: int | None = None
    tscc_is_offset_timing: bool | None = None
    ts_active_connection: bool | None = None
    aloha_mask: int | None = None
    service_function: int | None = None
    nrand_wait: int | None = None
    tscc_reg_required: bool | None = None
    tscc_backoff: int | None = None
    system_identity_code: int | None = None
    announcement_type: int | None = None
    broadcast_params_bits: str = ""
    raw_data_hex: str = ""

    def to_extra(self) -> dict:
        out = {
            "last_block": self.last_block,
            "protect_flag": self.protect_flag,
            "csbko_value": self.csbko_value,
            "csbko_name": self.csbko_name,
            "fid_value": self.fid_value,
            "fid_name": self.fid_name,
            "crc_value": self.crc,
            "source_address": self.source_address,
            "target_address": self.target_address,
            "bs_address": self.bs_address,
        }
        optional = {
            "answer_response_value": self.answer_response_value,
            "answer_response_name": self.answer_response_name,
            "additional_information_field": self.additional_information_field,
            "source_type": self.source_type,
            "service_type_value": self.service_type_value,
            "service_type_name": self.service_type_name,
            "reason_code": self.reason_code,
            "csbk_content_follows_preambles": self.csbk_content_follows_preambles,
            "target_address_is_individual": self.target_address_is_individual,
            "blocks_to_follow": self.blocks_to_follow,
            "sync_age": self.sync_age,
            "generation": self.generation,
            "leader_identifier": self.leader_identifier,
            "new_leader": self.new_leader,
            "leader_dynamic_identifier": self.leader_dynamic_identifier,
            "channel_timing_opcode": self.channel_timing_opcode,
            "source_identifier": self.source_identifier,
            "source_dynamic_identifier": self.source_dynamic_identifier,
            "tsccas_support": self.tsccas_support,
            "site_timeslot_synchronized": self.site_timeslot_synchronized,
            "document_version_control": self.document_version_control,
            "tscc_is_offset_timing": self.tscc_is_offset_timing,
            "ts_active_connection": self.ts_active_connection,
            "aloha_mask": self.aloha_mask,
            "service_function": self.service_function,
            "nrand_wait": self.nrand_wait,
            "tscc_reg_required": self.tscc_reg_required,
            "tscc_backoff": self.tscc_backoff,
            "system_identity_code": self.system_identity_code,
            "announcement_type": self.announcement_type,
        }
        out.update({k: v for k, v in optional.items() if v is not None and v != ""})
        if self.service_options is not None:
            out["service_options_value"] = self.service_options_value
            out["service_options"] = self.service_options.to_extra()
        if self.broadcast_params_bits:
            out["broadcast_params_bits"] = self.broadcast_params_bits
        if self.raw_data_hex:
            out["raw_data_hex"] = self.raw_data_hex
        return out


_FLCO_NAMES = {
    0x00: "GroupVoiceChannelUser",
    0x03: "UnitToUnitVoiceChannelUser",
    0x04: "TalkerAliasHeader",
    0x05: "TalkerAliasBlock1",
    0x06: "TalkerAliasBlock2",
    0x07: "TalkerAliasBlock3",
    0x08: "GPSInfo",
    0x30: "TerminatorDataLinkControl",
}

_FID_EXACT_NAMES = {
    0x00: "StandardizedFID",
    0x01: "ReservedForFutureStandardization",
    0x04: "FlydeMicroLtd",
    0x05: "ProdElSpa",
    0x06: "TridentMicroSystems",
    0x07: "RadiodataGmbh",
    0x08: "HytScienceTech",
    0x09: "AselsanElektronik",
    0x0A: "KirisunCommunications",
    0x0B: "DmrAssociationLtd",
    0x10: "MotorolaLtd",
    0x13: "ElectronicMarketingCompany",
    0x1C: "ElectronicMarketingCompany2",
    0x20: "JvcKenwood",
    0x33: "RadioActivity",
    0x3C: "RadioActivity2",
    0x58: "TaitElectronicsLtd",
    0x68: "HytScienceTech2",
    0x77: "VertexStandard",
    0x80: "ReservedForFutureMFID",
}

_CSBKO_NAMES = {
    0x04: "UnitToUnitVoiceServiceRequest",
    0x05: "UnitToUnitVoiceServiceAnswerResponse",
    0x07: "ChannelTimingCSBK",
    0x08: "HyteraIPSCSync",
    0x19: "AlohaPDUsForRandomAccessProtocol",
    0x1A: "UnifiedDataTransportOutboundHeader",
    0x1B: "UnifiedDataTransportInboundHeader",
    0x1C: "Ahoy",
    0x1E: "AckvitationPDU",
    0x1F: "RandomAccessServiceRequest",
    0x20: "AcknowledgementResponseOutboundTSCC",
    0x21: "AcknowledgementResponseInboundTSCC",
    0x22: "AcknowledgementResponseOutboundPayload",
    0x23: "AcknowledgementResponseInboundPayload",
    0x24: "UnifiedDataTransportForDGNAOutboundHeader",
    0x25: "UnifiedDataTransportForDGNAInboundHeader",
    0x26: "NegativeAcknowledgementResponse",
    0x28: "AnnouncementPDUsWithoutResponse",
    0x2A: "Maintenance",
    0x2E: "Clear",
    0x2F: "Protect",
    0x30: "PrivateVoiceChannelGrant",
    0x31: "TalkgroupVoiceChannelGrant",
    0x32: "PrivateBroadcastVoiceChannelGrant",
    0x33: "PrivateDataChannelGrantSingleItem",
    0x34: "TalkgroupDataChannelGrantSingleItem",
    0x35: "DuplexPrivateVoiceChannelGrant",
    0x36: "DuplexPrivateDataChannelGrant",
    0x37: "PrivateDataChannelGrantMultiItem",
    0x38: "BSOutboundActivation",
    0x39: "MovePDUs",
    0x3D: "PreambleCSBK",
}

_SUPPORTED_CSBKOS = {
    "BSOutboundActivation",
    "UnitToUnitVoiceServiceRequest",
    "UnitToUnitVoiceServiceAnswerResponse",
    "NegativeAcknowledgementResponse",
    "PreambleCSBK",
    "ChannelTimingCSBK",
    "HyteraIPSCSync",
    "AnnouncementPDUsWithoutResponse",
    "AlohaPDUsForRandomAccessProtocol",
}

_POSITION_ERROR_NAMES = {
    0x0: "LessThan2m",
    0x1: "LessThan20m",
    0x2: "LessThan200m",
    0x3: "LessThan2km",
    0x4: "LessThan20km",
    0x5: "LessThan200km",
    0x6: "MoreThan200km",
    0x7: "PositionErrorNotKnown",
}

_TALKER_ALIAS_FORMAT_NAMES = {
    0x0: "SevenBitCharacters",
    0x1: "ISOEightBitCharacters",
    0x2: "UnicodeUTF8",
    0x3: "UnicodeUTF16LE",
}

_ANSWER_RESPONSE_NAMES = {
    0x20: "Proceed",
    0x21: "Deny",
}


def _feature_set_name(value: int) -> str:
    if value in _FID_EXACT_NAMES:
        return _FID_EXACT_NAMES[value]
    if 0x01 <= value < 0x04:
        return "ReservedForFutureStandardization"
    if 0x04 <= value < 0x80:
        return "FlydeMicroLtd"
    return "ReservedForFutureMFID"


def parse_service_options(bits: bitarray) -> ServiceOptions:
    if len(bits) != 8:
        raise ValueError(f"Service Options expects 8 bits, got {len(bits)}")
    return ServiceOptions(
        value=ba2int(bits),
        emergency=bool(bits[0]),
        privacy=bool(bits[1]),
        reserved=ba2int(bits[2:4]),
        broadcast=bool(bits[4]),
        open_voice_call_mode=bool(bits[5]),
        priority=ba2int(bits[6:8]),
    )


def _bits_hex(bits: bitarray) -> str:
    return bits.tobytes().hex()


def parse_embedded_signalling(bits: bitarray) -> EmbeddedSignalling:
    if len(bits) != 16:
        raise ValueError(f"Embedded Signalling expects 16 bits, got {len(bits)}")
    lcss_value = ba2int(bits[5:7])
    return EmbeddedSignalling(
        colour_code=ba2int(bits[0:4]),
        preemption_and_power_control_indicator=int(bits[4]),
        link_control_start_stop=LCSS(lcss_value),
        emb_parity=ba2int(bits[7:16]),
        emb_parity_ok=qr_16_7_6_check(bits),
    )


def parse_full_link_control(bits: bitarray) -> FullLinkControl:
    if len(bits) not in (77, 96):
        raise ValueError(f"Full Link Control expects 77 or 96 bits, got {len(bits)}")

    flco_value = ba2int(bits[2:8])
    if flco_value not in _FLCO_NAMES:
        raise ValueError(f"Unsupported FLCO value {flco_value}")
    flco_name = _FLCO_NAMES[flco_value]
    fid_value = ba2int(bits[8:16])
    crc = ba2int(bits[72:96] if len(bits) >= 96 else bits[72:77])

    kwargs: dict[str, object] = {}
    if flco_name == "GroupVoiceChannelUser":
        service_options = parse_service_options(bits[16:24])
        kwargs["call_type"] = "group"
        kwargs["service_options_value"] = service_options.value
        kwargs["service_options"] = service_options
        kwargs["group_address"] = ba2int(bits[24:48])
        kwargs["source_address"] = ba2int(bits[48:72])
    elif flco_name == "UnitToUnitVoiceChannelUser":
        service_options = parse_service_options(bits[16:24])
        kwargs["call_type"] = "unit_to_unit"
        kwargs["service_options_value"] = service_options.value
        kwargs["service_options"] = service_options
        kwargs["target_address"] = ba2int(bits[24:48])
        kwargs["source_address"] = ba2int(bits[48:72])
    elif flco_name == "GPSInfo":
        position_error_value = ba2int(bits[20:23])
        kwargs["call_type"] = "gps"
        kwargs["position_error_value"] = position_error_value
        kwargs["position_error_name"] = _POSITION_ERROR_NAMES.get(
            position_error_value,
            f"UnknownPositionError0x{position_error_value:X}",
        )
        kwargs["longitude"] = (360 / 2**25) * ba2int(bits[23:48], signed=True)
        kwargs["latitude"] = (180 / 2**24) * ba2int(bits[48:72], signed=True)
    elif flco_name == "TalkerAliasHeader":
        fmt = ba2int(bits[16:18])
        kwargs["call_type"] = "talker_alias"
        kwargs["talker_alias_format_value"] = fmt
        kwargs["talker_alias_format_name"] = _TALKER_ALIAS_FORMAT_NAMES.get(fmt, "")
        kwargs["talker_alias_data_length"] = ba2int(bits[18:23])
        kwargs["talker_alias_data_msb"] = bool(bits[23])
        kwargs["talker_alias_data_hex"] = _bits_hex(bits[24:72])
        kwargs["talker_alias_data_bits"] = bits[23:72].to01()
    elif flco_name in {"TalkerAliasBlock1", "TalkerAliasBlock2", "TalkerAliasBlock3"}:
        kwargs["call_type"] = "talker_alias"
        kwargs["talker_alias_data_hex"] = _bits_hex(bits[16:72])
        kwargs["talker_alias_data_bits"] = bits[16:72].to01()
    else:
        raise ValueError(f"Unsupported FLCO {flco_name}")

    return FullLinkControl(
        protect_flag=bool(bits[0]),
        reserved_bit=int(bits[1]),
        flco_value=flco_value,
        flco_name=flco_name,
        fid_value=fid_value,
        fid_name=_feature_set_name(fid_value),
        crc=crc,
        **kwargs,
    )


def parse_csbk(bits: bitarray) -> CSBK:
    if len(bits) < 96:
        raise ValueError(f"CSBK expects at least 96 bits, got {len(bits)}")

    csbko_value = ba2int(bits[2:8])
    if csbko_value not in _CSBKO_NAMES:
        raise ValueError(f"Unsupported CSBKO value {csbko_value}")
    csbko_name = _CSBKO_NAMES[csbko_value]
    if csbko_name not in _SUPPORTED_CSBKOS:
        raise NotImplementedError(f"Unsupported CSBKO {csbko_name}")

    kwargs: dict[str, object] = {}
    if csbko_name == "BSOutboundActivation":
        kwargs["bs_address"] = ba2int(bits[32:56])
        kwargs["source_address"] = ba2int(bits[56:80])
    elif csbko_name in {
        "UnitToUnitVoiceServiceRequest",
        "PreambleCSBK",
    }:
        if csbko_name == "UnitToUnitVoiceServiceRequest":
            service_options = parse_service_options(bits[16:24])
            kwargs["service_options_value"] = service_options.value
            kwargs["service_options"] = service_options
        else:
            kwargs["csbk_content_follows_preambles"] = not bool(bits[16])
            kwargs["target_address_is_individual"] = not bool(bits[17])
            kwargs["blocks_to_follow"] = ba2int(bits[24:32])
        kwargs["target_address"] = ba2int(bits[32:56])
        kwargs["source_address"] = ba2int(bits[56:80])
    elif csbko_name == "UnitToUnitVoiceServiceAnswerResponse":
        service_options = parse_service_options(bits[16:24])
        answer = ba2int(bits[24:32])
        kwargs["service_options_value"] = service_options.value
        kwargs["service_options"] = service_options
        kwargs["answer_response_value"] = answer
        kwargs["answer_response_name"] = _ANSWER_RESPONSE_NAMES.get(answer, f"Unknown0x{answer:02X}")
        kwargs["target_address"] = ba2int(bits[32:56])
        kwargs["source_address"] = ba2int(bits[56:80])
    elif csbko_name == "NegativeAcknowledgementResponse":
        service_type_value = ba2int(bits[18:24])
        kwargs["additional_information_field"] = int(bits[16])
        kwargs["source_type"] = int(bits[17])
        kwargs["service_type_value"] = service_type_value
        kwargs["service_type_name"] = _CSBKO_NAMES.get(
            service_type_value,
            f"UnknownCSBKO0x{service_type_value:02X}",
        )
        kwargs["reason_code"] = ba2int(bits[24:32])
        kwargs["source_address"] = ba2int(bits[32:56])
        kwargs["target_address"] = ba2int(bits[56:80])
    elif csbko_name == "ChannelTimingCSBK":
        kwargs["sync_age"] = ba2int(bits[16:27])
        kwargs["generation"] = ba2int(bits[27:32])
        kwargs["leader_identifier"] = ba2int(bits[32:52])
        kwargs["new_leader"] = bool(bits[52])
        kwargs["leader_dynamic_identifier"] = ba2int(bits[53:55])
        kwargs["channel_timing_opcode"] = ba2int(bitarray([bits[55], bits[79]], endian="big"))
        kwargs["source_identifier"] = ba2int(bits[56:76])
        kwargs["source_dynamic_identifier"] = ba2int(bits[77:79])
    elif csbko_name == "HyteraIPSCSync":
        kwargs["raw_data_hex"] = _bits_hex(bits[16:80])
    elif csbko_name == "AlohaPDUsForRandomAccessProtocol":
        kwargs["tsccas_support"] = bool(bits[17])
        kwargs["site_timeslot_synchronized"] = bool(bits[18])
        kwargs["document_version_control"] = ba2int(bits[19:22])
        kwargs["tscc_is_offset_timing"] = bool(bits[22])
        kwargs["ts_active_connection"] = bool(bits[23])
        kwargs["aloha_mask"] = ba2int(bits[24:29])
        kwargs["service_function"] = ba2int(bits[29:31])
        kwargs["nrand_wait"] = ba2int(bits[31:35])
        kwargs["tscc_reg_required"] = bool(bits[35])
        kwargs["tscc_backoff"] = ba2int(bits[36:40])
        kwargs["system_identity_code"] = ba2int(bits[40:56])
        kwargs["target_address"] = ba2int(bits[56:80])
    elif csbko_name == "AnnouncementPDUsWithoutResponse":
        kwargs["announcement_type"] = ba2int(bits[16:21])
        kwargs["tscc_reg_required"] = bool(bits[35])
        kwargs["tscc_backoff"] = ba2int(bits[36:40])
        kwargs["system_identity_code"] = ba2int(bits[40:56])
        kwargs["broadcast_params_bits"] = (bits[21:35] + bits[56:80]).to01()

    fid_value = ba2int(bits[8:16])
    return CSBK(
        last_block=bool(bits[0]),
        protect_flag=bool(bits[1]),
        csbko_value=csbko_value,
        csbko_name=csbko_name,
        fid_value=fid_value,
        fid_name=_feature_set_name(fid_value),
        crc=ba2int(bits[80:96]),
        **kwargs,
    )
