-- M1 notifications migration: WhatsApp number + email/WhatsApp prefs + log.
-- Rerunnable: scripts/migrate_auth_db.py checks INFORMATION_SCHEMA before
-- applying each statement, so running this file twice is safe.

ALTER TABLE users
  ADD COLUMN whatsapp_e164 VARCHAR(20) NULL;

CREATE TABLE IF NOT EXISTS notification_prefs (
  user_id BIGINT UNSIGNED NOT NULL,
  email_enabled TINYINT(1) NOT NULL DEFAULT 1,
  whatsapp_enabled TINYINT(1) NOT NULL DEFAULT 0,
  assets JSON NULL,
  min_probability DECIMAL(4,3) NOT NULL DEFAULT 0.700,
  daily_summary TINYINT(1) NOT NULL DEFAULT 0,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (user_id),
  CONSTRAINT fk_notif_prefs_user FOREIGN KEY (user_id)
    REFERENCES users (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS notification_log (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  user_id BIGINT UNSIGNED NOT NULL,
  channel ENUM('email','whatsapp') NOT NULL,
  type VARCHAR(32) NOT NULL DEFAULT 'high_range',
  asset VARCHAR(16) NOT NULL,
  dedupe_key VARCHAR(128) NOT NULL,
  status VARCHAR(16) NOT NULL DEFAULT 'sent',
  provider_msg_id VARCHAR(128) NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_notif_dedupe (dedupe_key, channel),
  KEY ix_notif_user_created (user_id, created_at),
  CONSTRAINT fk_notif_log_user FOREIGN KEY (user_id)
    REFERENCES users (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
