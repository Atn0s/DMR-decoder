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


@dataclass(frozen=True)
class FullLinkControl:
    protect_flag: bool
    flco_value: int
    flco_name: str
    fid_value: int
    fid_name: str
    crc: int
    source_address: int = 0
    group_address: int = 0
    target_address: int = 0


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


def _feature_set_name(value: int) -> str:
    if value in _FID_EXACT_NAMES:
        return _FID_EXACT_NAMES[value]
    if 0x01 <= value < 0x04:
        return "ReservedForFutureStandardization"
    if 0x04 <= value < 0x80:
        return "FlydeMicroLtd"
    return "ReservedForFutureMFID"


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

    kwargs: dict[str, int] = {}
    if flco_name == "GroupVoiceChannelUser":
        kwargs["group_address"] = ba2int(bits[24:48])
        kwargs["source_address"] = ba2int(bits[48:72])
    elif flco_name == "UnitToUnitVoiceChannelUser":
        kwargs["target_address"] = ba2int(bits[24:48])
        kwargs["source_address"] = ba2int(bits[48:72])
    elif flco_name in {
        "GPSInfo",
        "TalkerAliasHeader",
        "TalkerAliasBlock1",
        "TalkerAliasBlock2",
        "TalkerAliasBlock3",
    }:
        pass
    else:
        raise ValueError(f"Unsupported FLCO {flco_name}")

    return FullLinkControl(
        protect_flag=bool(bits[0]),
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

    kwargs: dict[str, int] = {}
    if csbko_name == "BSOutboundActivation":
        kwargs["source_address"] = ba2int(bits[56:80])
    elif csbko_name in {
        "UnitToUnitVoiceServiceRequest",
        "UnitToUnitVoiceServiceAnswerResponse",
        "PreambleCSBK",
    }:
        kwargs["target_address"] = ba2int(bits[32:56])
        kwargs["source_address"] = ba2int(bits[56:80])
    elif csbko_name == "NegativeAcknowledgementResponse":
        kwargs["source_address"] = ba2int(bits[32:56])
        kwargs["target_address"] = ba2int(bits[56:80])
    elif csbko_name == "AlohaPDUsForRandomAccessProtocol":
        kwargs["target_address"] = ba2int(bits[56:80])

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

