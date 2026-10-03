"""SMTP email sender (Gmail SMTP for dev, Resend/SES later via same env shape)."""

from __future__ import annotations

import os
import smtplib
import uuid
from email.message import EmailMessage


class SmtpEmailSender:
    """Thin smtplib wrapper implementing the EmailSender protocol."""

    def __init__(
        self,
        host: str,
        port: int = 587,
        username: str = "",
        password: str = "",
        from_address: str = "",
        use_tls: bool = True,
        timeout_seconds: int = 15,
    ) -> None:
        if not host:
            raise ValueError("SMTP_HOST is required")
        if not from_address:
            raise ValueError("SMTP_FROM is required")
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.from_address = from_address
        self.use_tls = use_tls
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_env(cls) -> "SmtpEmailSender":
        host = os.getenv("SMTP_HOST", "")
        try:
            port = int(os.getenv("SMTP_PORT", "587"))
        except ValueError:
            raise ValueError("SMTP_PORT must be an integer") from None
        use_tls = os.getenv("SMTP_USE_TLS", "true").lower() in {"1", "true", "yes"}
        return cls(
            host=host,
            port=port,
            username=os.getenv("SMTP_USER", ""),
            password=os.getenv("SMTP_PASS", ""),
            from_address=os.getenv("SMTP_FROM", os.getenv("SMTP_USER", "")),
            use_tls=use_tls,
        )

    def send(self, to: str, subject: str, text_body: str, html_body: str | None = None) -> str:
        if not to or "@" not in to:
            raise ValueError("A valid recipient email is required")
        message = EmailMessage()
        message["From"] = self.from_address
        message["To"] = to
        message["Subject"] = subject
        message.set_content(text_body)
        if html_body:
            message.add_alternative(html_body, subtype="html")
        if self.use_tls:
            with smtplib.SMTP(self.host, self.port, timeout=self.timeout_seconds) as client:
                client.starttls()
                if self.username:
                    client.login(self.username, self.password)
                client.send_message(message)
        else:
            with smtplib.SMTP(self.host, self.port, timeout=self.timeout_seconds) as client:
                if self.username:
                    client.login(self.username, self.password)
                client.send_message(message)
        return f"smtp-{uuid.uuid4().hex[:12]}"
