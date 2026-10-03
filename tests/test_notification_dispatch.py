"""Dispatcher + sender unit tests (no network, no MySQL)."""

from datetime import datetime, timezone

from src.notifications import dispatcher, rules
from src.notifications.smtp_email import SmtpEmailSender
from src.notifications.whatsapp_cloud import WhatsAppCloudSender


class FakeDB:
    def __init__(self, rows):
        self.rows = rows
        self.sent_keys = set()
        self.inserts = []

    def fetch_all(self, query, params=()):
        return self.rows

    def fetch_one(self, query, params=()):
        key = (params[0], params[1])
        return {"id": 1} if key in self.sent_keys else None

    def execute(self, query, params=()):
        # INSERT INTO notification_log (user_id, channel, ..., dedupe_key, ...)
        self.inserts.append(params)
        self.sent_keys.add((params[3], params[1]))
        return 1


class FakeEmail:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def send(self, to, subject, text_body, html_body=None):
        self.calls.append((to, subject))
        if self.fail:
            raise RuntimeError("smtp down")
        return "smtp-1"


class FakeWhatsApp:
    def __init__(self):
        self.calls = []

    def send(self, to_e164, template_name, params, body_text=None):
        self.calls.append((to_e164, template_name, params))
        return "wamid.1"


def _rows():
    return [
        {
            "id": 1, "email": "a@example.com", "whatsapp_e164": "+919876543210",
            "email_enabled": 1, "whatsapp_enabled": 1,
            "assets": '["RELIANCE"]', "min_probability": 0.70, "daily_summary": 0,
        },
        {
            "id": 2, "email": "b@example.com", "whatsapp_e164": None,
            "email_enabled": 0, "whatsapp_enabled": 0,
            "assets": None, "min_probability": 0.70, "daily_summary": 0,
        },
    ]


def _pred(proba=0.85):
    return {
        "asset": "RELIANCE",
        "prediction": 1,
        "probability_high_range": proba,
        "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
    }


# Monday 09:30 IST — inside the 09:15-15:30 session gate.
IN_SESSION = datetime(2026, 8, 10, 4, 0, tzinfo=timezone.utc)


def test_dispatch_sends_email_and_whatsapp_once():
    db = FakeDB(_rows())
    email, wa = FakeEmail(), FakeWhatsApp()
    pred = _pred()
    summary = dispatcher.send_for_snapshot(
        {"RELIANCE": pred}, db=db, email_sender=email, whatsapp_sender=wa, moment=IN_SESSION
    )
    assert summary["sent_email"] == 1
    assert summary["sent_whatsapp"] == 1
    # rerun with the same prediction_timestamp dedupes via notification_log
    summary2 = dispatcher.send_for_snapshot(
        {"RELIANCE": pred}, db=db, email_sender=email, whatsapp_sender=wa, moment=IN_SESSION
    )
    assert summary2["sent_email"] == 0 and summary2["sent_whatsapp"] == 0
    assert len(email.calls) == 1 and len(wa.calls) == 1


def test_dispatch_dry_run_sends_nothing():
    db = FakeDB(_rows())
    email, wa = FakeEmail(), FakeWhatsApp()
    summary = dispatcher.send_for_snapshot(
        {"RELIANCE": _pred()}, db=db, email_sender=email, whatsapp_sender=wa, moment=IN_SESSION, dry_run=True
    )
    assert summary["sent_email"] == 1  # counted
    assert email.calls == [] and wa.calls == []
    assert db.inserts == []


def test_dispatch_failure_isolated_and_below_threshold_skipped():
    db = FakeDB(_rows())
    email, wa = FakeEmail(fail=True), FakeWhatsApp()
    summary = dispatcher.send_for_snapshot(
        {"RELIANCE": _pred(proba=0.85)}, db=db, email_sender=email, whatsapp_sender=wa, moment=IN_SESSION
    )
    assert summary["failed"] == 1
    assert summary["sent_whatsapp"] == 1  # whatsapp still goes out
    summary2 = dispatcher.send_for_snapshot(
        {"RELIANCE": _pred(proba=0.10)},
        db=db,
        email_sender=FakeEmail(),
        whatsapp_sender=FakeWhatsApp(),
        moment=IN_SESSION,
    )
    assert summary2["sent_email"] == 0 and summary2["sent_whatsapp"] == 0


def test_smtp_requires_host_and_from(monkeypatch):
    import pytest

    monkeypatch.delenv("SMTP_HOST", raising=False)
    with pytest.raises(ValueError):
        SmtpEmailSender.from_env()


def test_whatsapp_requires_token(monkeypatch):
    import pytest

    monkeypatch.delenv("WHATSAPP_TOKEN", raising=False)
    monkeypatch.setenv("WHATSAPP_PHONE_ID", "123")
    with pytest.raises(ValueError):
        WhatsAppCloudSender.from_env()


def test_whatsapp_api_number_strips_plus():
    assert WhatsAppCloudSender.to_api_number("+919876543210") == "919876543210"


def test_email_subject_mentions_research_not_advice():
    subject, body = rules.format_email("RELIANCE", 0.8, datetime.now(timezone.utc).isoformat())
    assert "research" in subject.lower()
    assert "not trading advice" in body.lower()
