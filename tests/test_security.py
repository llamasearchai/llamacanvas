"""
Tests for security features in LlamaCanvas.

This module contains tests for security features such as
input validation, authentication, authorization, and data sanitization.
"""

import base64
import hashlib
import json
import os
import re
import tempfile
from unittest.mock import MagicMock, call, patch

import pytest

from llama_canvas.security import (
    Authenticator,
    DataSanitizer,
    InputValidator,
    PermissionManager,
    SecurityManager,
    decrypt_data,
    encrypt_data,
    generate_token,
    hash_password,
    sanitize_input,
    scan_file_content,
    validate_file_path,
    validate_image_data,
    verify_password,
    verify_token,
)


class TestSecurityManager:
    """Tests for the SecurityManager class."""

    def test_init(self):
        """Test initialization of SecurityManager."""
        manager = SecurityManager()

        # Test default properties
        assert manager.enabled is True
        assert manager.authenticator is not None
        assert manager.input_validator is not None
        assert manager.data_sanitizer is not None
        assert manager.permission_manager is not None

    def test_authenticate_user(self):
        """Test user authentication."""
        manager = SecurityManager()

        # Mock authenticator
        manager.authenticator = MagicMock()
        manager.authenticator.authenticate.return_value = True

        # Test authentication
        result = manager.authenticate_user("username", "password")

        # Should delegate to authenticator
        assert manager.authenticator.authenticate.called
        assert manager.authenticator.authenticate.call_args[0] == (
            "username",
            "password",
        )

        # Should return result from authenticator
        assert result is True

    def test_validate_input(self):
        """Test input validation."""
        manager = SecurityManager()

        # Mock validator
        manager.input_validator = MagicMock()
        manager.input_validator.validate_input.return_value = (True, None)

        # Test validation
        result, error = manager.validate_input("test_input", "text")

        # Should delegate to validator
        assert manager.input_validator.validate_input.called
        assert manager.input_validator.validate_input.call_args[0] == (
            "test_input",
            "text",
        )

        # Should return result from validator
        assert result is True
        assert error is None

    def test_sanitize_input(self):
        """Test input sanitization."""
        manager = SecurityManager()

        # Mock sanitizer
        manager.data_sanitizer = MagicMock()
        manager.data_sanitizer.sanitize.return_value = "sanitized_input"

        # Test sanitization
        result = manager.sanitize_input("<script>alert('xss')</script>")

        # Should delegate to sanitizer
        assert manager.data_sanitizer.sanitize.called
        assert (
            manager.data_sanitizer.sanitize.call_args[0][0]
            == "<script>alert('xss')</script>"
        )

        # Should return result from sanitizer
        assert result == "sanitized_input"

    def test_check_permission(self):
        """Test permission checking."""
        manager = SecurityManager()

        # Mock permission manager
        manager.permission_manager = MagicMock()
        manager.permission_manager.check_permission.return_value = True

        # Test permission checking
        result = manager.check_permission("user123", "edit_image")

        # Should delegate to permission manager
        assert manager.permission_manager.check_permission.called
        assert manager.permission_manager.check_permission.call_args[0] == (
            "user123",
            "edit_image",
        )

        # Should return result from permission manager
        assert result is True


class TestAuthenticator:
    """Tests for the Authenticator class."""

    def test_init(self):
        """Test initialization of Authenticator."""
        auth = Authenticator()

        # Test default properties
        assert auth.users == {}
        assert auth.token_expiry == 3600  # Default 1 hour

    def test_register_user(self):
        """Test user registration."""
        auth = Authenticator()

        with patch("llama_canvas.security.hash_password") as mock_hash:
            # Mock password hashing
            mock_hash.return_value = "hashed_password"

            # Register user
            auth.register_user("testuser", "password123")

            # Should hash password
            assert mock_hash.called
            assert mock_hash.call_args[0][0] == "password123"

            # Should store user
            assert "testuser" in auth.users
            assert auth.users["testuser"]["password"] == "hashed_password"
            assert "created_at" in auth.users["testuser"]

    def test_authenticate(self):
        """Test user authentication."""
        auth = Authenticator()

        # Add test user
        auth.users = {
            "testuser": {"password": "hashed_password", "created_at": 12345678}
        }

        with patch("llama_canvas.security.verify_password") as mock_verify:
            # Mock password verification
            mock_verify.return_value = True

            # Authenticate user
            result = auth.authenticate("testuser", "password123")

            # Should verify password
            assert mock_verify.called
            assert mock_verify.call_args[0] == ("password123", "hashed_password")

            # Should return True for successful authentication
            assert result is True

            # Test with invalid username
            result = auth.authenticate("nonexistent", "password123")

            # Should return False for invalid username
            assert result is False

            # Test with invalid password
            mock_verify.return_value = False
            result = auth.authenticate("testuser", "wrong_password")

            # Should return False for invalid password
            assert result is False

    def test_generate_token(self):
        """Test token generation."""
        auth = Authenticator()

        with patch("llama_canvas.security.generate_token") as mock_generate:
            # Mock token generation
            mock_generate.return_value = "test_token"

            # Generate token
            token = auth.generate_token("testuser")

            # Should call token generation
            assert mock_generate.called
            assert mock_generate.call_args[0][0] == "testuser"

            # Should return token
            assert token == "test_token"

    def test_verify_token(self):
        """Test token verification."""
        auth = Authenticator()

        with patch("llama_canvas.security.verify_token") as mock_verify:
            # Mock token verification
            mock_verify.return_value = ("testuser", True)

            # Verify token
            username, valid = auth.verify_token("test_token")

            # Should call token verification
            assert mock_verify.called
            assert mock_verify.call_args[0][0] == "test_token"

            # Should return verification result
            assert username == "testuser"
            assert valid is True


class TestPasswordFunctions:
    """Tests for password hashing and verification functions."""

    def test_hash_password(self):
        """Test password hashing."""
        # Hash password
        hashed = hash_password("test_password")

        # Should return a string
        assert isinstance(hashed, str)

        # Different passwords should have different hashes
        hashed2 = hash_password("different_password")
        assert hashed != hashed2

        # Same password should have different hashes (due to salt)
        hashed3 = hash_password("test_password")
        assert hashed != hashed3

    def test_verify_password(self):
        """Test password verification."""
        # Hash password
        hashed = hash_password("test_password")

        # Verify correct password
        assert verify_password("test_password", hashed) is True

        # Verify incorrect password
        assert verify_password("wrong_password", hashed) is False


class TestTokenFunctions:
    """Tests for token generation and verification functions."""

    def test_generate_token(self):
        """Test token generation."""
        # Generate token
        token = generate_token("testuser")

        # Should return a string
        assert isinstance(token, str)

        # Different usernames should generate different tokens
        token2 = generate_token("otheruser")
        assert token != token2

    def test_verify_token(self):
        """Test token verification."""
        # Generate token
        token = generate_token("testuser")

        # Verify valid token
        username, valid = verify_token(token)

        # Should extract correct username
        assert username == "testuser"
        assert valid is True

        # Verify invalid token
        username, valid = verify_token("invalid_token")

        # Should fail verification
        assert valid is False

        # Verify expired token
        with patch("llama_canvas.security.time") as mock_time:
            # Mock current time to be far in the future
            mock_time.time.return_value = 9999999999

            username, valid = verify_token(token)

            # Should fail verification
            assert valid is False


class TestInputValidator:
    """Tests for the InputValidator class."""

    def test_init(self):
        """Test initialization of InputValidator."""
        validator = InputValidator()

        # Test default properties
        assert validator.validators == {}
        assert validator.max_length == 1024  # Default max length

    def test_register_validator(self):
        """Test registering custom validators."""
        validator = InputValidator()

        # Define test validation function
        def validate_email(input_str):
            if "@" in input_str:
                return True, None
            return False, "Invalid email format"

        # Register validator
        validator.register_validator("email", validate_email)

        # Validator should be registered
        assert "email" in validator.validators
        assert validator.validators["email"] is validate_email

    def test_validate_input(self):
        """Test input validation."""
        validator = InputValidator()

        # Register test validators
        def validate_email(input_str):
            if "@" in input_str:
                return True, None
            return False, "Invalid email format"

        def validate_number(input_str):
            if input_str.isdigit():
                return True, None
            return False, "Must be a number"

        validator.register_validator("email", validate_email)
        validator.register_validator("number", validate_number)

        # Test valid email
        valid, error = validator.validate_input("test@example.com", "email")
        assert valid is True
        assert error is None

        # Test invalid email
        valid, error = validator.validate_input("invalid_email", "email")
        assert valid is False
        assert error == "Invalid email format"

        # Test valid number
        valid, error = validator.validate_input("12345", "number")
        assert valid is True
        assert error is None

        # Test invalid number
        valid, error = validator.validate_input("abc", "number")
        assert valid is False
        assert error == "Must be a number"

        # Test with unknown validator type
        valid, error = validator.validate_input("test", "unknown")
        assert valid is False
        assert "Unknown validator" in error

        # Test input length validation
        validator.max_length = 5
        valid, error = validator.validate_input("too_long", "email")
        assert valid is False
        assert "exceeds maximum length" in error


class TestDataSanitizer:
    """Tests for the DataSanitizer class."""

    def test_init(self):
        """Test initialization of DataSanitizer."""
        sanitizer = DataSanitizer()

        # Test default properties
        assert sanitizer.html_allowed_tags == []

    def test_sanitize_html(self):
        """Test HTML sanitization."""
        sanitizer = DataSanitizer()

        # Allow some HTML tags
        sanitizer.html_allowed_tags = ["b", "i", "a"]

        # Test with script tag (should be removed)
        html = "<b>Bold</b> <script>alert('xss')</script> <i>Italic</i>"
        result = sanitizer.sanitize_html(html)

        # Script tag should be removed
        assert "<script>" not in result
        assert "alert" not in result

        # Allowed tags should remain
        assert "<b>Bold</b>" in result
        assert "<i>Italic</i>" in result

    def test_sanitize_filename(self):
        """Test filename sanitization."""
        sanitizer = DataSanitizer()

        # Test with invalid characters
        filename = "../../malicious/file.exe"
        result = sanitizer.sanitize_filename(filename)

        # Path traversal should be removed
        assert "../" not in result

        # Test with spaces and special characters
        filename = "my file (1).txt"
        result = sanitizer.sanitize_filename(filename)

        # Should replace spaces and handle special characters
        assert " " not in result
        assert "(" not in result
        assert ")" not in result

    def test_sanitize(self):
        """Test general input sanitization."""
        sanitizer = DataSanitizer()

        # Test with different input types
        with patch.object(sanitizer, "sanitize_html") as mock_html, patch.object(
            sanitizer, "sanitize_filename"
        ) as mock_filename:

            # Mock sanitization methods
            mock_html.return_value = "sanitized_html"
            mock_filename.return_value = "sanitized_filename"

            # Test HTML sanitization
            result = sanitizer.sanitize(
                "<script>alert('xss')</script>", input_type="html"
            )
            assert mock_html.called
            assert result == "sanitized_html"

            # Test filename sanitization
            result = sanitizer.sanitize("../file.exe", input_type="filename")
            assert mock_filename.called
            assert result == "sanitized_filename"

            # Test default sanitization (text)
            result = sanitizer.sanitize("Normal text with special chars: <>&")

            # Should escape special characters
            assert "<" not in result
            assert ">" not in result
            assert "&" not in result


class TestPermissionManager:
    """Tests for the PermissionManager class."""

    def test_init(self):
        """Test initialization of PermissionManager."""
        manager = PermissionManager()

        # Test default properties
        assert manager.roles == {}
        assert manager.user_roles == {}

    def test_define_role(self):
        """Test defining roles and permissions."""
        manager = PermissionManager()

        # Define role
        manager.define_role("editor", ["view_image", "edit_image", "save_image"])

        # Role should be defined
        assert "editor" in manager.roles
        assert "view_image" in manager.roles["editor"]
        assert "edit_image" in manager.roles["editor"]
        assert "save_image" in manager.roles["editor"]

        # Define another role
        manager.define_role("viewer", ["view_image"])

        # Both roles should exist
        assert "viewer" in manager.roles
        assert len(manager.roles["viewer"]) == 1
        assert "view_image" in manager.roles["viewer"]

    def test_assign_role(self):
        """Test assigning roles to users."""
        manager = PermissionManager()

        # Define roles
        manager.define_role("editor", ["edit_image"])
        manager.define_role("viewer", ["view_image"])

        # Assign role to user
        manager.assign_role("user1", "editor")

        # User should have role
        assert "user1" in manager.user_roles
        assert manager.user_roles["user1"] == "editor"

        # Assign different role
        manager.assign_role("user2", "viewer")

        # Both users should have appropriate roles
        assert manager.user_roles["user1"] == "editor"
        assert manager.user_roles["user2"] == "viewer"

        # Change user's role
        manager.assign_role("user1", "viewer")

        # User's role should be updated
        assert manager.user_roles["user1"] == "viewer"

    def test_check_permission(self):
        """Test checking user permissions."""
        manager = PermissionManager()

        # Define roles
        manager.define_role("admin", ["view_image", "edit_image", "delete_image"])
        manager.define_role("editor", ["view_image", "edit_image"])
        manager.define_role("viewer", ["view_image"])

        # Assign roles
        manager.assign_role("admin_user", "admin")
        manager.assign_role("editor_user", "editor")
        manager.assign_role("viewer_user", "viewer")

        # Check permissions for admin
        assert manager.check_permission("admin_user", "view_image") is True
        assert manager.check_permission("admin_user", "edit_image") is True
        assert manager.check_permission("admin_user", "delete_image") is True

        # Check permissions for editor
        assert manager.check_permission("editor_user", "view_image") is True
        assert manager.check_permission("editor_user", "edit_image") is True
        assert manager.check_permission("editor_user", "delete_image") is False

        # Check permissions for viewer
        assert manager.check_permission("viewer_user", "view_image") is True
        assert manager.check_permission("viewer_user", "edit_image") is False
        assert manager.check_permission("viewer_user", "delete_image") is False

        # Check for unknown user
        assert manager.check_permission("unknown_user", "view_image") is False

        # Check for unknown permission
        assert manager.check_permission("admin_user", "unknown_permission") is False


class TestValidationFunctions:
    """Tests for validation utility functions."""

    def test_validate_file_path(self):
        """Test file path validation."""
        # Test valid path
        valid, error = validate_file_path("images/test.jpg")
        assert valid is True
        assert error is None

        # Test path traversal attack
        valid, error = validate_file_path("../../etc/passwd")
        assert valid is False
        assert "Path traversal" in error

        # Test invalid extension
        valid, error = validate_file_path(
            "image.exe", allowed_extensions=[".jpg", ".png"]
        )
        assert valid is False
        assert "extension not allowed" in error

        # Test absolute path
        valid, error = validate_file_path("/etc/hosts", allow_absolute=False)
        assert valid is False
        assert "Absolute paths not allowed" in error

    def test_validate_image_data(self):
        """Test image data validation."""
        # Create test image data
        valid_image_data = base64.b64encode(b"fake_image_data").decode("utf-8")

        with patch("llama_canvas.security.PILImage") as mock_pil:
            # Mock image opening
            mock_image = MagicMock()
            mock_pil.open.return_value = mock_image

            # Test valid image
            mock_image.format = "JPEG"
            mock_image.verify.return_value = None

            valid, error = validate_image_data(valid_image_data)
            assert valid is True
            assert error is None

            # Test invalid image format
            mock_image.format = "UNKNOWN"

            valid, error = validate_image_data(
                valid_image_data, allowed_formats=["JPEG", "PNG"]
            )
            assert valid is False
            assert "format not allowed" in error

            # Test corrupt image
            mock_image.verify.side_effect = Exception("Invalid image data")

            valid, error = validate_image_data(valid_image_data)
            assert valid is False
            assert "Invalid image" in error


class TestScanFile:
    """Tests for file scanning functionality."""

    def test_scan_file_content(self):
        """Test scanning file content for security issues."""
        # Create temporary test file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # Write benign content
            temp_file.write(b"This is a test file with normal content.")
            temp_file_path = temp_file.name

        try:
            # Test scanning benign file
            issues = scan_file_content(temp_file_path)
            assert issues == []

            # Create file with malicious content
            with open(temp_file_path, "w") as f:
                f.write("#!/bin/bash\nrm -rf /\n")

            # Test scanning malicious file
            issues = scan_file_content(temp_file_path)
            assert len(issues) > 0
            assert any("suspicious shell command" in issue for issue in issues)

            # Create file with potential XSS
            with open(temp_file_path, "w") as f:
                f.write("<script>alert('XSS')</script>")

            # Test scanning file with XSS
            issues = scan_file_content(temp_file_path)
            assert len(issues) > 0
            assert any("potential XSS" in issue for issue in issues)

        finally:
            # Clean up test file
            os.unlink(temp_file_path)


class TestEncryption:
    """Tests for data encryption and decryption."""

    def test_encrypt_decrypt(self):
        """Test encrypting and decrypting data."""
        # Original data
        data = "Sensitive information"

        # Encrypt data
        encrypted = encrypt_data(data)

        # Should return bytes
        assert isinstance(encrypted, bytes)

        # Encrypted data should be different from original
        assert encrypted != data
        assert encrypted != data.encode("utf-8")

        # Decrypt data
        decrypted = decrypt_data(encrypted)

        # Should return original data
        assert decrypted == data

        # Test with different data
        data2 = "Different sensitive information"
        encrypted2 = encrypt_data(data2)

        # Different data should result in different ciphertext
        assert encrypted != encrypted2

        # Decryption should work for each
        assert decrypt_data(encrypted) == data
        assert decrypt_data(encrypted2) == data2


class TestIntegrationTests:
    """Integration tests for security features."""

    def test_security_workflow(self):
        """Test complete security workflow."""
        # Create security manager
        manager = SecurityManager()

        # Register user
        manager.authenticator.register_user("testuser", "secure_password")

        # Define roles and permissions
        manager.permission_manager.define_role("editor", ["view_image", "edit_image"])
        manager.permission_manager.assign_role("testuser", "editor")

        # User input with potential XSS
        user_input = "<script>alert('XSS')</script>Hello, world!"

        # Validate and sanitize input
        is_valid, error = manager.validate_input(user_input, "text")
        assert is_valid is True  # Text input is valid but needs sanitizing

        sanitized = manager.sanitize_input(user_input)
        assert "<script>" not in sanitized
        assert "Hello, world!" in sanitized

        # Authenticate user
        auth_result = manager.authenticate_user("testuser", "secure_password")
        assert auth_result is True

        # Generate token
        token = manager.authenticator.generate_token("testuser")

        # Verify token
        username, valid = manager.authenticator.verify_token(token)
        assert username == "testuser"
        assert valid is True

        # Check permissions
        assert manager.check_permission("testuser", "view_image") is True
        assert manager.check_permission("testuser", "edit_image") is True
        assert manager.check_permission("testuser", "delete_image") is False

        # Test file path validation
        valid_path, error = validate_file_path("images/user_upload.jpg")
        assert valid_path is True

        invalid_path, error = validate_file_path("../system/config.ini")
        assert invalid_path is False


if __name__ == "__main__":
    pytest.main()
