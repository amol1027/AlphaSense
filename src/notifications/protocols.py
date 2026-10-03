"""Replaceable provider interfaces for email + WhatsApp.

Downstream code must depend on these protocols, never on smtplib /
requests / SDK clients directly, so Gmail SMTP can be swapped for
Resend/SES and Meta Cloud can be swapped for MSG91/Twilio without
touching rules or dispatch.
"""

from __future__ import annotations

from typing import Protocol


class EmailSender(Protocol):
    def send(self, to: str, subject: str, text_body: str, html_body: str | None = None) -> str:
        """Send an email, return the provider message id (or local id)."""
        ...


class WhatsAppSender(Protocol):
    def send(self, to_e164: str, template_name: str, params: list[str], body_text: str | None = None) -> str:
        """Send a WhatsApp template message, return provider message id."""
        ...
