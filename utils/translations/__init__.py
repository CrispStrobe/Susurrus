"""Translation catalogs, one module per locale.

Kept out of ``utils/i18n.py`` so the lookup logic stays readable next to a
catalog of several hundred strings. ``utils.i18n`` assembles these into its
public ``TRANSLATIONS`` mapping.

Adding a locale: create ``<code>.py`` exporting a ``STRINGS`` dict and add it
to ``CATALOGS`` below. ``tests/unit/test_i18n.py`` enforces that every locale
defines every key, so a partial catalog fails the suite rather than silently
falling back to English.
"""

from . import de, en

CATALOGS = {
    "en": en.STRINGS,
    "de": de.STRINGS,
}

#: Human-readable locale names for the GUI language selector.
LOCALE_NAMES = {
    "en": "English",
    "de": "Deutsch",
}

__all__ = ["CATALOGS", "LOCALE_NAMES"]
