"""Meta WhatsApp Cloud API sender (free 1,000 service conversations/month).

Docs: https://developers.facebook.com/docs/whatsapp/cloud-api
Setup: Meta Business app -> WhatsApp -> API setup -> PHONE_NUMBER_ID +
permanent access token -> approved `volatility_alert` utility template.
"""

from __future__ import annotations

import os

import requests


class WhatsAppCloudSender:
    """Thin requests wrapper implementing the WhatsAppSender protocol."""

    def __init__(
        self,
        token: str,
        phone_number_id: str,
        api_version: str = "v21.0",
        language: str = "en",
        timeout_seconds: int = 15,
    ) -> None:
        if not token:
            raise ValueError("WHATSAPP_TOKEN is required")
        if not phone_number_id:
            raise ValueError("WHATSAPP_PHONE_ID is required")
        self.token = token
        self.phone_number_id = phone_number_id
        self.api_version = api_version.strip().lstrip("/")
        self.language = language
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_env(cls) -> "WhatsAppCloudSender":
        return cls(
            token=os.getenv("WHATSAPP_TOKEN", ""),
            phone_number_id=os.getenv("WHATSAPP_PHONE_ID", ""),
            api_version=os.getenv("WHATSAPP_API_VERSION", "v21.0"),
            language=os.getenv("WHATSAPP_LANGUAGE", "en"),
        )

    @staticmethod
    def to_api_number(e164: str) -> str:
        """Cloud API wants digits without the leading '+'."""
        cleaned = e164.strip().replace(" ", "").replace("-", "")
        return cleaned[1:] if cleaned.startswith("+") else cleaned

    def send(self, to_e164: str, template_name: str, params: list[str], body_text: str | None = None) -> str:
        to = self.to_api_number(to_e164)
        if not to.isdigit() or len(to) < 8:
            raise ValueError("A valid E.164 WhatsApp number is required")
        url = f"https://graph.facebook.com/{self.api_version}/{self.phone_number_id}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "template",
            "template": {
                "name": template_name,
                "language": {"code": self.language},
                "components": [
                    {"type": "body", "parameters": [{"type": "text", "text": p} for p in params]},
                ],
            },
        }
        response = requests.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
            timeout=self.timeout_seconds,
        )
        if response.status_code >= 400:
            detail = response.text[:300]
            raise RuntimeError(f"WhatsApp send failed ({response.status_code}): {detail}")
        try:
            messages = response.json().get("messages", [])
            if messages and messages[0].get("id"):
                return str(messages[0]["id"])
        except ValueError:
            pass
        return f"wa-{to[-4:]}"
