"""Minimal i18n — string lookup with English fallback.

Catalogs live in ``utils/translations/`` (one module per locale) so this file
stays small next to several hundred strings. ``TRANSLATIONS`` is re-exported
for callers and tests that want the raw mapping.
"""

import logging

from utils.translations import CATALOGS, LOCALE_NAMES

logger = logging.getLogger(__name__)

_current_locale = "en"

#: {locale_code: {key: text}} — the assembled catalog.
TRANSLATIONS = CATALOGS

__all__ = [
    "TRANSLATIONS",
    "LOCALE_NAMES",
    "t",
    "set_locale",
    "get_locale",
    "locale_name",
    "available_locales",
    "detect_system_locale",
]


def set_locale(locale):
    """Set the active locale (e.g. 'en', 'de'). Unknown locales fall back."""
    global _current_locale
    if locale in TRANSLATIONS:
        _current_locale = locale
    else:
        logger.warning("Unknown locale '%s', falling back to 'en'", locale)
        _current_locale = "en"
    return _current_locale


def get_locale():
    """Return the current locale."""
    return _current_locale


def t(key, locale=None):
    """Translate a key. Falls back to English, then to the key itself.

    Args:
        key: The translation key.
        locale: Override the active locale for this lookup — used where the
            output language is decided per-call rather than per-session, e.g.
            a spoken disclosure that must match the synthesis language.
    """
    strings = TRANSLATIONS.get(locale or _current_locale, TRANSLATIONS["en"])
    if key in strings:
        return strings[key]
    # Fallback to English
    return TRANSLATIONS["en"].get(key, key)


def available_locales():
    """Return the list of available locale codes."""
    return list(TRANSLATIONS.keys())


def locale_name(code):
    """Return the human-readable name for a locale code."""
    return LOCALE_NAMES.get(code, code)


def detect_system_locale(default="en"):
    """Best-effort detection of the user's language from the environment.

    Only the language part is used: ``de_DE.UTF-8`` and ``de_AT`` both map to
    ``de``. Returns *default* when nothing matches a catalog we ship.
    """
    import locale as _locale
    import os

    candidates = []
    for var in ("LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"):
        value = os.environ.get(var)
        if value:
            candidates.extend(value.split(":"))

    try:
        system, _encoding = _locale.getdefaultlocale()
        if system:
            candidates.append(system)
    except (ValueError, TypeError):  # pragma: no cover — platform dependent
        pass

    for candidate in candidates:
        code = candidate.split(".")[0].split("_")[0].strip().lower()
        if code in TRANSLATIONS:
            return code
    return default
