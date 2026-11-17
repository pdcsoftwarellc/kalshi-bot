"""Regression checks for credential isolation, startup failures, and auth."""
import base64
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa

import config
from bot import TradingBot
from integration.kalshi_client import KalshiClient
from integration.preflight import startup_errors
from scripts import download_data


@pytest.mark.asyncio
@pytest.mark.parametrize("demo", [True, False])
async def test_paper_uses_public_production_even_with_live_config(monkeypatch, demo):
    monkeypatch.setattr(config, "HYBRID_PRICING", True)
    monkeypatch.setattr(config, "PROD_KALSHI_API_KEY", "production-secret")
    monkeypatch.setattr(config, "KALSHI_PRIVATE_KEY_PATH", "/does/not/exist.pem")
    client = TradingBot(paper=True, demo=demo)._create_kalshi_client()

    def respond(request):
        assert str(request.url).startswith(config.KALSHI_PROD_BASE)
        assert not any(name.startswith("kalshi-access") for name in request.headers)
        assert request.method == "GET"
        return httpx.Response(200, json={"markets": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as transport:
        client._client = transport
        assert await client.get_markets("KXATPMATCH") == []


@pytest.mark.parametrize("key_type", ["rsa", "ed25519"])
def test_signature_verifies_and_excludes_query(tmp_path, key_type):
    key = (rsa.generate_private_key(public_exponent=65537, key_size=2048)
           if key_type == "rsa" else ed25519.Ed25519PrivateKey.generate())
    pem_path = tmp_path / "test.pem"
    pem_path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))
    client = KalshiClient("test-key", str(pem_path))
    ts, signature = client._sign("get", "/trade-api/v2/portfolio/balance?subaccount=1")
    message = f"{ts}GET/trade-api/v2/portfolio/balance".encode()
    if key_type == "rsa":
        key.public_key().verify(base64.b64decode(signature), message,
                                padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                                            salt_length=padding.PSS.DIGEST_LENGTH),
                                hashes.SHA256())
    else:
        key.public_key().verify(base64.b64decode(signature), message)


@pytest.mark.asyncio
async def test_missing_auth_fails_before_any_request():
    client = KalshiClient("", "")
    client._client = Mock()
    with pytest.raises(ValueError, match="credentials"):
        await client.get_balance()
    client._client.request.assert_not_called()


def test_demo_does_not_fall_back_to_production_credentials(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setattr(config, "HYBRID_PRICING", False)
    monkeypatch.setattr(config, "DEMO_KALSHI_API_KEY", "")
    monkeypatch.setattr(config, "DEMO_KALSHI_PRIVATE_KEY_PATH", "")
    monkeypatch.setattr(config, "KALSHI_API_KEY", "production-key")
    monkeypatch.setattr(config, "KALSHI_PRIVATE_KEY_PATH", "/production.pem")
    errors = startup_errors(paper=False, demo=True)
    assert any("Set DEMO_KALSHI_API_KEY" in message for message in errors)
    client = TradingBot(paper=False, demo=True)._create_kalshi_client()
    assert client._api_key == ""
    assert client._private_key is None


def test_production_flag_does_not_route_to_hybrid_demo(monkeypatch):
    monkeypatch.setattr(config, "HYBRID_PRICING", True)
    monkeypatch.setattr(config, "PROD_KALSHI_API_KEY", "hybrid-read-key")
    monkeypatch.setattr(config, "KALSHI_API_KEY", "production-key")
    monkeypatch.setattr(config, "KALSHI_PRIVATE_KEY_PATH", "")
    client = TradingBot(paper=False, demo=False)._create_kalshi_client()
    assert client._base == config.KALSHI_PROD_BASE
    assert client._api_key == "production-key"


@pytest.mark.parametrize("setting,value", [
    ("MIN_CONFIDENCE", float("nan")),
    ("MAX_TOTAL_EXPOSURE", float("inf")),
    ("MAX_CONTRACTS_PER_SIGNAL", 0),
    ("WEB_PORT", 70000),
])
def test_invalid_operating_limits_are_reported(monkeypatch, tmp_path, setting, value):
    monkeypatch.setattr(config, "ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setattr(config, setting, value)
    assert any(setting in message for message in startup_errors())


def test_missing_setup_is_reported_together_without_writes(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "API_TENNIS_KEY", "")
    monkeypatch.setattr(config, "ARTIFACTS_DIR", str(tmp_path))
    errors = startup_errors()
    assert any("API_TENNIS_KEY" in message for message in errors)
    assert any("gbdt_model.pkl, calibrator.pkl" in message for message in errors)
    assert list(tmp_path.iterdir()) == []


def test_download_failure_exits_nonzero(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(download_data, "DATA_RAW_DIR", str(tmp_path))
    monkeypatch.setattr(download_data.sys, "argv", ["download_data.py", "--repos", "tennis_atp"])
    monkeypatch.setattr(download_data.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(returncode=128, stdout="", stderr="Repository not found"))
    with pytest.raises(SystemExit) as error:
        download_data.main()
    assert error.value.code == 1
    output = capsys.readouterr()
    assert "Download failed" in output.err
    assert "Done." not in output.out
