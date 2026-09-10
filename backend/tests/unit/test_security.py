"""Authentication, storage-key handling and the rights gate."""

from __future__ import annotations

import time
import uuid

import pytest

from app.core.errors import AuthenticationError, AuthorizationStatusError, ValidationFailure
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    generate_api_key,
    hash_api_key,
    hash_password,
    verify_password,
)
from app.models.enums import (
    PROCESSABLE_AUTHORIZATION,
    PUBLISHABLE_AUTHORIZATION,
    AuthorizationStatus,
)
from app.models.video import Video
from app.providers.storage.local_storage import (
    LocalStorageProvider,
    _sanitize_key,
    sign_key,
    verify_signature,
)
from app.services.media import assert_processable
from app.services.storage_paths import safe_filename


class TestPasswords:
    def test_round_trip(self):
        encoded = hash_password("correct-horse-battery")
        assert verify_password("correct-horse-battery", encoded)
        assert not verify_password("wrong-password", encoded)

    def test_hash_is_salted(self):
        assert hash_password("same-password-x") != hash_password("same-password-x")

    def test_plaintext_never_appears_in_the_hash(self):
        assert "correct-horse-battery" not in hash_password("correct-horse-battery")

    def test_short_passwords_are_rejected(self):
        with pytest.raises(ValueError):
            hash_password("short")

    @pytest.mark.parametrize("garbage", ["", "not-a-hash", "a$b$c", "pbkdf2_sha256$x$y$z"])
    def test_malformed_hashes_fail_closed(self, garbage):
        assert verify_password("anything", garbage) is False


class TestTokens:
    def test_access_token_round_trip(self):
        subject = str(uuid.uuid4())
        payload = decode_token(create_access_token(subject))
        assert payload["sub"] == subject
        assert payload["type"] == "access"

    def test_a_refresh_token_is_not_an_access_token(self):
        token = create_refresh_token(str(uuid.uuid4()))
        with pytest.raises(AuthenticationError):
            decode_token(token, expected_type="access")
        assert decode_token(token, expected_type="refresh")["type"] == "refresh"

    def test_tampered_tokens_are_rejected(self):
        token = create_access_token(str(uuid.uuid4()))
        head, payload, signature = token.split(".")
        with pytest.raises(AuthenticationError):
            decode_token(f"{head}.{payload}.{signature[:-4]}abcd")

    def test_garbage_is_rejected(self):
        with pytest.raises(AuthenticationError):
            decode_token("not.a.token")

    def test_tokens_carry_a_unique_id(self):
        subject = str(uuid.uuid4())
        first = decode_token(create_access_token(subject))
        second = decode_token(create_access_token(subject))
        assert first["jti"] != second["jti"]


class TestApiKeys:
    def test_only_the_digest_is_derivable(self):
        raw, digest = generate_api_key()
        assert raw.startswith("vca_")
        assert digest == hash_api_key(raw)
        assert raw not in digest

    def test_keys_are_unique(self):
        assert generate_api_key()[0] != generate_api_key()[0]


class TestStorageKeys:
    @pytest.mark.parametrize(
        "key",
        [
            "../../etc/passwd",
            "users/../../secrets",
            "a/../../b",
            "/../root",
            "..",
        ],
    )
    def test_traversal_is_refused(self, key):
        with pytest.raises(ValidationFailure):
            _sanitize_key(key)

    def test_windows_drive_letters_are_refused(self):
        with pytest.raises(ValidationFailure):
            _sanitize_key("C:/Windows/System32/config")

    def test_empty_keys_are_refused(self):
        for key in ("", "/", "///"):
            with pytest.raises(ValidationFailure):
                _sanitize_key(key)

    def test_normal_keys_pass_through(self):
        assert _sanitize_key("/users/abc/videos/1/source/x.mp4") == (
            "users/abc/videos/1/source/x.mp4"
        )
        assert _sanitize_key("a\\b\\c.mp4") == "a/b/c.mp4"

    def test_provider_refuses_to_escape_its_root(self, tmp_path):
        storage = LocalStorageProvider(root=tmp_path, base_url="http://x/media")
        with pytest.raises(ValidationFailure):
            storage.exists("../outside.txt")

    def test_upload_size_limit_is_enforced(self, tmp_path):
        import io

        storage = LocalStorageProvider(root=tmp_path, base_url="http://x/media")
        with pytest.raises(ValidationFailure):
            storage.put_stream("a/b.bin", io.BytesIO(b"x" * 5000), max_bytes=1000)
        # The partial file must not be left behind.
        assert not (tmp_path / "a" / "b.bin").exists()

    def test_uploaded_filenames_are_neutralised(self):
        assert safe_filename("../../evil.mp4") == "evil.mp4"
        assert safe_filename("my video (1).mp4") == "my_video_1_.mp4"
        assert safe_filename("") == "file"
        assert "/" not in safe_filename("a/b/c.mp4")


class TestSignedUrls:
    def test_a_valid_signature_verifies(self):
        expires = int(time.time()) + 600
        assert verify_signature("a/b.mp4", expires, sign_key("a/b.mp4", expires))

    def test_an_expired_signature_fails(self):
        expires = int(time.time()) - 5
        assert not verify_signature("a/b.mp4", expires, sign_key("a/b.mp4", expires))

    def test_a_signature_does_not_transfer_to_another_key(self):
        expires = int(time.time()) + 600
        assert not verify_signature("other/file.mp4", expires, sign_key("a/b.mp4", expires))

    def test_a_forged_signature_fails(self):
        assert not verify_signature("a/b.mp4", int(time.time()) + 600, "forged")


class TestRightsGate:
    """The rights model is the product's central constraint, so it is asserted
    directly rather than only through the API."""

    def test_processable_set_excludes_unknown_and_refused(self):
        assert AuthorizationStatus.UNKNOWN not in PROCESSABLE_AUTHORIZATION
        assert AuthorizationStatus.NOT_AUTHORIZED not in PROCESSABLE_AUTHORIZATION
        assert AuthorizationStatus.USER_OWNED in PROCESSABLE_AUTHORIZATION
        assert AuthorizationStatus.USER_UPLOADED in PROCESSABLE_AUTHORIZATION

    def test_publishable_set_excludes_unknown_and_refused(self):
        assert AuthorizationStatus.UNKNOWN not in PUBLISHABLE_AUTHORIZATION
        assert AuthorizationStatus.NOT_AUTHORIZED not in PUBLISHABLE_AUTHORIZATION

    @pytest.mark.parametrize(
        "status", [AuthorizationStatus.UNKNOWN, AuthorizationStatus.NOT_AUTHORIZED]
    )
    def test_unauthorized_media_cannot_be_processed(self, status):
        video = Video(
            user_id=uuid.uuid4(),
            source="DISCOVERY",
            title="Someone else's video",
            authorization_status=status,
        )
        with pytest.raises(AuthorizationStatusError) as excinfo:
            assert_processable(video)
        assert "authorization_status" in str(excinfo.value)

    @pytest.mark.parametrize("status", sorted(PROCESSABLE_AUTHORIZATION, key=str))
    def test_authorized_media_passes_the_gate(self, status):
        video = Video(
            user_id=uuid.uuid4(),
            source="UPLOAD",
            title="My own video",
            authorization_status=status,
        )
        assert_processable(video)  # must not raise
        assert video.is_processable is True
