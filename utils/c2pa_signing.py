"""C2PA Content Credentials signing for audio files (EU AI Act Art. 50(2)).

Two implementations are supported, tried in this order:

1. ``c2pa_audio`` — the CrispASR-adjacent binding
   (https://github.com/CrispStrobe/c2pa-audio). Not published on PyPI, so it
   is only present where someone installed it from source.
2. ``c2pa`` — the official c2pa-python bindings from PyPI. This is the one a
   normal ``pip install`` can actually get, and it signs WAV, MP3 and FLAC.

Before this module supported (2), the whole C2PA layer was dead on every
Python-native TTS path: the only import was a package that cannot be
installed from an index, so ``sign_wav_file()`` returned False forever while
the docs advertised C2PA as a shipped provenance layer. Art. 50(2) has no
"unless the optional dependency was unavailable" clause.

**Signing identity.** C2PA signing needs an X.509 credential. If you supply
``--c2pa-cert`` / ``--c2pa-key``, those are used. Otherwise this module
generates a *local* CA + end-entity chain once and caches it under
``~/.local/share/susurrus/c2pa/``. That identity is self-issued: it proves
the file has not been altered since Susurrus signed it, and it proves nothing
about who you are. For a credential that carries identity, pass a CA-issued
certificate. (c2pa-rs rejects a bare self-signed leaf, which is why a
two-cert chain is generated rather than a single certificate.)

**Timestamping.** Signing is offline by default. An RFC 3161 timestamp makes
a signature outlive its certificate, but it also means a network round-trip
per synthesis to a third party — the wrong default for a local-first tool.
Set ``SUSURRUS_C2PA_TSA`` to a TSA URL to opt in.
"""

import logging
import os

logger = logging.getLogger(__name__)

#: IPTC digital source type for content created by a generative model. This
#: is the vocabulary term that makes a manifest say "AI-generated" in the way
#: Art. 50(2) means, rather than merely recording that a tool touched a file.
TRAINED_ALGORITHMIC_MEDIA = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"

#: Audio extensions c2pa-python can embed a manifest into.
_SIGNABLE_EXTENSIONS = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
}

_backend = None
_backend_name = None
_backend_checked = False


def _detect_backend():
    """Resolve the C2PA implementation to use. Returns (name, handle)."""
    global _backend, _backend_name, _backend_checked
    if _backend_checked:
        return _backend_name, _backend
    _backend_checked = True

    try:
        from c2pa_audio import C2paAudio

        _backend = C2paAudio()
        _backend_name = "c2pa-audio"
        logger.info("C2PA backend: c2pa-audio (v%s)", _backend.version)
        return _backend_name, _backend
    except (ImportError, OSError, AttributeError) as e:
        logger.debug("c2pa-audio not available: %s", e)

    try:
        import c2pa

        _backend = c2pa
        _backend_name = "c2pa-python"
        logger.info("C2PA backend: c2pa-python (SDK %s)", c2pa.sdk_version())
        return _backend_name, _backend
    except (ImportError, OSError) as e:
        logger.debug("c2pa-python not available: %s", e)

    _backend, _backend_name = None, None
    return None, None


def _get_c2pa():
    """Lazy-load the C2PA signer. Returns None if unavailable."""
    return _detect_backend()[1]


def is_available():
    """Check if C2PA signing is available."""
    return _detect_backend()[1] is not None


def backend_name():
    """Return the active C2PA implementation name, or None."""
    return _detect_backend()[0]


def _resolve_pem(value):
    """Accept either an inline PEM string or a path to a PEM file.

    The CLI (``--c2pa-cert``) and GUI file pickers supply paths, while the
    signing APIs want the PEM text itself.
    """
    if not value:
        return None
    if "-----BEGIN" in value:
        return value
    try:
        with open(value, "r", encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        logger.warning("Could not read PEM file %s: %s", value, e)
        return None


# --------------------------------------------------------------------------
# Local signing identity
# --------------------------------------------------------------------------


def identity_dir():
    """Return the directory holding the generated local signing identity."""
    xdg = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
    d = os.path.join(xdg, "susurrus", "c2pa")
    os.makedirs(d, mode=0o700, exist_ok=True)
    return d


def _identity_paths():
    d = identity_dir()
    return os.path.join(d, "signer-chain.pem"), os.path.join(d, "signer-key.pem")


def _build_local_identity():
    """Generate a CA + end-entity chain for local signing.

    c2pa-rs validates the signing certificate's profile and rejects a bare
    self-signed leaf, so this issues a proper (if locally-rooted) chain:
    a CA certificate that signs an end-entity certificate carrying the
    emailProtection EKU the C2PA cert profile expects.
    """
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    now = datetime.datetime.now(datetime.timezone.utc)
    not_before = now - datetime.timedelta(days=1)
    not_after = now + datetime.timedelta(days=3650)

    def _name(cn):
        return x509.Name(
            [
                x509.NameAttribute(NameOID.COMMON_NAME, cn),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Susurrus"),
            ]
        )

    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = _name("Susurrus Local Root CA")
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_before)
        .not_valid_after(not_after)
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False
        )
        .sign(ca_key, hashes.SHA256())
    )

    leaf_key = ec.generate_private_key(ec.SECP256R1())
    leaf_cert = (
        x509.CertificateBuilder()
        .subject_name(_name("Susurrus Local Signer"))
        .issuer_name(ca_name)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_before)
        .not_valid_after(not_after)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION]), critical=False
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(leaf_key.public_key()), critical=False
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    chain_pem = (
        leaf_cert.public_bytes(serialization.Encoding.PEM)
        + ca_cert.public_bytes(serialization.Encoding.PEM)
    ).decode("ascii")
    key_pem = leaf_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode("ascii")
    return chain_pem, key_pem


def ensure_local_identity():
    """Return (chain_pem, key_pem) for the cached local identity.

    Generates and caches it on first use. Returns (None, None) if
    ``cryptography`` is unavailable.
    """
    chain_path, key_path = _identity_paths()
    if os.path.isfile(chain_path) and os.path.isfile(key_path):
        try:
            with open(chain_path, "r", encoding="utf-8") as f:
                chain = f.read()
            with open(key_path, "r", encoding="utf-8") as f:
                key = f.read()
            if chain and key:
                return chain, key
        except OSError as e:
            logger.warning("Could not read cached C2PA identity: %s", e)

    try:
        chain, key = _build_local_identity()
    except ImportError:
        logger.warning(
            "C2PA signing needs a certificate. Install 'cryptography' for an "
            "auto-generated local identity, or pass --c2pa-cert/--c2pa-key."
        )
        return None, None
    except Exception as e:
        logger.warning("Could not generate a local C2PA identity: %s", e)
        return None, None

    try:
        with open(chain_path, "w", encoding="utf-8") as f:
            f.write(chain)
        with open(key_path, "w", encoding="utf-8") as f:
            f.write(key)
        os.chmod(key_path, 0o600)
        logger.info("Generated a local C2PA signing identity in %s", identity_dir())
    except OSError as e:
        # An identity we cannot cache still signs this run.
        logger.warning("Could not cache the C2PA identity: %s", e)

    return chain, key


def _alg_and_callback(key_pem):
    """Return (C2paSigningAlg, callback) for *key_pem*, or (None, None).

    COSE wants ECDSA signatures as raw r||s of fixed width; ``cryptography``
    emits DER, so the two halves are unpacked and re-padded.
    """
    import c2pa
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
    from cryptography.hazmat.primitives.asymmetric import utils as asym_utils

    try:
        key = serialization.load_pem_private_key(key_pem.encode("utf-8"), password=None)
    except Exception as e:
        logger.warning("Could not load the C2PA private key: %s", e)
        return None, None

    if isinstance(key, ec.EllipticCurvePrivateKey):
        by_size = {
            256: (c2pa.C2paSigningAlg.ES256, hashes.SHA256(), 32),
            384: (c2pa.C2paSigningAlg.ES384, hashes.SHA384(), 48),
            521: (c2pa.C2paSigningAlg.ES512, hashes.SHA512(), 66),
        }
        entry = by_size.get(key.curve.key_size)
        if entry is None:
            logger.warning("Unsupported EC curve for C2PA: %s", key.curve.name)
            return None, None
        alg, digest, width = entry

        def _sign(data):
            der = key.sign(data, ec.ECDSA(digest))
            r, s = asym_utils.decode_dss_signature(der)
            return r.to_bytes(width, "big") + s.to_bytes(width, "big")

        return alg, _sign

    if isinstance(key, rsa.RSAPrivateKey):

        def _sign(data):
            return key.sign(
                data,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32),
                hashes.SHA256(),
            )

        return c2pa.C2paSigningAlg.PS256, _sign

    logger.warning("Unsupported C2PA key type: %s", type(key).__name__)
    return None, None


def _manifest(model=None, software="Susurrus"):
    """Build the manifest asserting that this audio was AI-generated."""
    from utils.ai_marking import AI_DISCLOSURE_TEXT

    generator = {"name": software}
    version = _version()
    if version:
        generator["version"] = version
    if model:
        generator["model"] = str(model)

    return {
        "claim_generator_info": [generator],
        "title": "AI-generated audio",
        "assertions": [
            {
                "label": "c2pa.actions",
                "data": {
                    "actions": [
                        {
                            "action": "c2pa.created",
                            "digitalSourceType": TRAINED_ALGORITHMIC_MEDIA,
                            "softwareAgent": software,
                        }
                    ]
                },
            },
            {
                "label": "stds.schema-org.CreativeWork",
                "data": {
                    "@context": "https://schema.org",
                    "@type": "CreativeWork",
                    "description": AI_DISCLOSURE_TEXT,
                },
            },
        ],
    }


def _version():
    try:
        from __init__ import __version__

        return __version__
    except ImportError:
        return None


def _sign_with_c2pa_python(path, mime, cert_pem, key_pem, model=None):
    """Sign *path* in place using c2pa-python. Returns True on success."""
    import c2pa

    if not cert_pem or not key_pem:
        cert_pem, key_pem = ensure_local_identity()
    if not cert_pem or not key_pem:
        return False

    alg, callback = _alg_and_callback(key_pem)
    if alg is None:
        return False

    tsa_url = os.environ.get("SUSURRUS_C2PA_TSA") or None

    tmp_path = f"{path}.c2pa.tmp{os.path.splitext(path)[1]}"
    try:
        signer = c2pa.Signer.from_callback(
            callback=callback, alg=alg, certs=cert_pem, tsa_url=tsa_url
        )
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            builder = c2pa.Builder(_manifest(model=model))
            builder.sign_file(path, tmp_path, signer)
            os.replace(tmp_path, path)
        finally:
            close = getattr(signer, "close", None)
            if callable(close):
                close()
    except Exception as e:
        logger.warning("C2PA signing failed for %s: %s", path, e)
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        return False

    logger.info("C2PA signed (%s): %s", mime, path)
    return True


def sign_audio_file(path, cert_pem=None, key_pem=None, model=None):
    """Sign an audio file with C2PA Content Credentials in-place.

    Args:
        path: Path to the audio file (WAV, MP3, FLAC or M4A).
        cert_pem: PEM certificate, inline or as a file path. Falls back to the
            cached local identity.
        key_pem: PEM private key, inline or as a file path.
        model: Model identifier recorded in the manifest.

    Returns:
        True if signed, False otherwise. Never raises.
    """
    name, backend = _detect_backend()
    if backend is None:
        logger.debug("C2PA unavailable; skipping signing for %s", path)
        return False

    if not path or not os.path.isfile(path):
        return False

    ext = os.path.splitext(path)[1].lower()
    mime = _SIGNABLE_EXTENSIONS.get(ext)
    if mime is None:
        logger.warning("C2PA signing skipped, unsupported container: %s", path)
        return False

    cert_pem = _resolve_pem(cert_pem)
    key_pem = _resolve_pem(key_pem)

    if name == "c2pa-python":
        return _sign_with_c2pa_python(path, mime, cert_pem, key_pem, model=model)

    # c2pa-audio path (WAV only)
    try:
        with open(path, "rb") as f:
            data = f.read()
        signed = backend.sign_wav(data, cert_pem=cert_pem, key_pem=key_pem)
        with open(path, "wb") as f:
            f.write(signed)
        logger.info("C2PA signed: %s", path)
        return True
    except Exception as e:
        logger.warning("C2PA signing failed for %s: %s", path, e)
        return False


def sign_wav_file(wav_path, cert_pem=None, key_pem=None, model=None):
    """Backwards-compatible alias for :func:`sign_audio_file`."""
    return sign_audio_file(wav_path, cert_pem=cert_pem, key_pem=key_pem, model=model)


def verify_audio_file(path):
    """Verify C2PA Content Credentials in an audio file.

    Returns:
        dict with ``valid``, ``signature_valid``, ``data_hash_valid``,
        ``assertions_valid`` and (c2pa-python only) ``ai_generated`` and
        ``validation_state``. None if no C2PA implementation is available.
    """
    name, backend = _detect_backend()
    if backend is None:
        return None

    if not path or not os.path.isfile(path):
        # "Cannot tell" — not "carries no credentials".
        return None

    if name == "c2pa-python":
        return _verify_with_c2pa_python(path)

    try:
        with open(path, "rb") as f:
            data = f.read()
        result = backend.verify_wav(data)
        return {
            "valid": result.valid,
            "signature_valid": result.signature_valid,
            "data_hash_valid": result.data_hash_valid,
            "assertions_valid": result.assertions_valid,
        }
    except Exception as e:
        logger.warning("C2PA verification failed for %s: %s", path, e)
        return None


def _verify_with_c2pa_python(path):
    """Read and validate a manifest with c2pa-python."""
    import json

    import c2pa

    try:
        reader = c2pa.Reader(path)
        try:
            state = str(reader.get_validation_state())
            manifests = json.loads(reader.json())
        finally:
            close = getattr(reader, "close", None)
            if callable(close):
                close()
    except Exception as e:
        # "no manifest" is the common case, not an error worth shouting about.
        logger.debug("No readable C2PA manifest in %s: %s", path, e)
        return {
            "valid": False,
            "signature_valid": False,
            "data_hash_valid": False,
            "assertions_valid": False,
            "ai_generated": False,
            "validation_state": "NoManifest",
        }

    valid = "invalid" not in state.lower()
    active = manifests.get("manifests", {}).get(manifests.get("active_manifest"), {})
    ai_generated = TRAINED_ALGORITHMIC_MEDIA in json.dumps(active.get("assertions", []))

    return {
        "valid": valid,
        "signature_valid": valid,
        "data_hash_valid": valid,
        "assertions_valid": valid,
        "ai_generated": ai_generated,
        "validation_state": state,
    }


def verify_wav_file(wav_path):
    """Backwards-compatible alias for :func:`verify_audio_file`."""
    return verify_audio_file(wav_path)


def _reset_cache_for_tests():
    """Clear the resolved-backend cache (test helper)."""
    global _backend, _backend_name, _backend_checked
    _backend = None
    _backend_name = None
    _backend_checked = False
