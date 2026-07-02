import dmr.cli as dmr_cli


def test_dmr_cli_uses_unified_scanner_entry(monkeypatch, capsys):
    calls = []

    monkeypatch.setattr(dmr_cli.os.path, "exists", lambda path: True)

    def fake_scan_file(path, freq_list, output_json, protocol_names):
        calls.append((path, freq_list, output_json, protocol_names))
        return []

    monkeypatch.setattr(dmr_cli.scanner, "scan_file", fake_scan_file)

    result = dmr_cli.main([
        "sample.rawiq",
        "--fo",
        "1250",
        "--json",
        "out.json",
    ])

    assert result == 0
    assert calls == [("sample.rawiq", [1250.0], "out.json", ["dmr"])]
    assert "=== sample.rawiq ===" in capsys.readouterr().out
