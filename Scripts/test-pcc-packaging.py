#!/usr/bin/env python3
"""Offline regression tests for PCC signing authorization checks."""
import copy
import datetime
import hashlib
import importlib.util
import pathlib
import plistlib
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("packager", pathlib.Path(__file__).with_name("package-pcc-app.py"))
packager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packager)


class PCCPackagingTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime.datetime(2026, 9, 12, tzinfo=datetime.timezone.utc)
        self.profile = {
            "ExpirationDate": datetime.datetime(2027, 1, 1),
            "Platform": ["OSX"], "ProvisionedDevices": ["test-mac"],
            "DeveloperCertificates": [b"fixture-certificate"],
            "Entitlements": {
                packager.PCC: True,
                packager.APP_ID: "PREFIX.com.scouzi1966.afm",
                packager.TEAM_ID: "TEAM",
                packager.APP_ATTEST: ["CDhash"],
                "com.apple.developer.foundation-model-adapter": True,
            },
        }

    def validate(self, profile=None):
        return packager.validate_profile(profile or self.profile, "com.scouzi1966.afm", self.now)

    def test_preserves_profile_identity_and_attestation_without_unrelated_claims(self):
        claims = self.validate()
        self.assertEqual(claims[packager.APP_ID], "PREFIX.com.scouzi1966.afm")
        self.assertEqual(claims[packager.TEAM_ID], "TEAM")
        self.assertEqual(claims[packager.APP_ATTEST], ["CDhash"])
        self.assertNotIn("com.apple.developer.foundation-model-adapter", claims)

    def test_rejects_vesta_and_wildcard_profiles(self):
        for identifier in ["TEAM.soprano.Vesta-mac", "TEAM.*"]:
            self.profile["Entitlements"][packager.APP_ID] = identifier
            with self.assertRaisesRegex(ValueError, "AFM-specific"):
                self.validate()

    def test_rejects_missing_pcc_expired_ios_and_distribution_profiles(self):
        variants = []
        missing = copy.deepcopy(self.profile)
        del missing["Entitlements"][packager.PCC]
        variants.append(missing)
        variants.append(dict(self.profile, ExpirationDate=datetime.datetime(2025, 1, 1)))
        variants.append(dict(self.profile, Platform=["iOS"]))
        variants.append(dict(self.profile, ProvisionsAllDevices=True))
        variants.append(dict(self.profile, ProvisionedDevices=[]))
        for profile in variants:
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                self.validate(profile)

    def test_identity_must_be_development_and_authorized_by_profile(self):
        fingerprint = hashlib.sha1(b"fixture-certificate").hexdigest().upper()
        name = "Apple Development: Test (TEAM)"
        listing = f'1) {fingerprint} "{name}"'
        self.assertEqual(packager.select_identity(self.profile, name, listing), fingerprint)
        self.assertEqual(packager.select_identity(self.profile, fingerprint, listing), fingerprint)
        for bad_listing in [listing.replace("Apple Development:", "Developer ID Application:"),
                            listing.replace(fingerprint, "0" * 40), "0 valid identities found"]:
            with self.assertRaises(ValueError):
                packager.select_identity(self.profile, fingerprint, bad_listing)

    def test_apple_silicon_registration_uses_provisioning_udid(self):
        packager.validate_device(self.profile, {"platform_UUID": "different", "provisioning_UDID": "test-mac"})
        with self.assertRaises(ValueError):
            packager.validate_device(self.profile, {"platform_UUID": "test-mac", "provisioning_UDID": "not-registered"})
        with self.assertRaises(ValueError):
            packager.validate_device(self.profile, {})

    def test_reads_linked_identity_including_trailing_otool_bytes(self):
        data = plistlib.dumps({"CFBundleIdentifier": "com.scouzi1966.afm"})
        words = [data[i:i + 4][::-1].hex() if len(data[i:i + 4]) == 4
                 else " ".join(f"{value:02x}" for value in data[i:i + 4])
                 for i in range(0, len(data), 4)]
        output = "0000000100000000 " + " ".join(words)
        with patch.object(packager, "run", return_value=output.encode()):
            self.assertEqual(packager.embedded_info("afm")["CFBundleIdentifier"], "com.scouzi1966.afm")


if __name__ == "__main__":
    unittest.main()
