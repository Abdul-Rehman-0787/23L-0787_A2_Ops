"""The text of the emails VisionGuard sends."""
from html import escape

brandName = "VisionGuard"


def buildCodeEmail(code, minutes=5):
    """Returns (subject, plain text, html) for the sign-in verification code."""
    subject = "Verify your " + brandName + " account"
    safeCode = escape(code)

    text = (
        "Verify your " + brandName + " account\n"
        "\n"
        "Hello there,\n"
        "\n"
        "Please enter the following verification code to complete your sign-in:\n"
        "\n"
        "    " + code + "\n"
        "\n"
        "This code will expire in " + str(minutes) + " minutes.\n"
        "\n"
        "Security tip: " + brandName + " will never call or message you to ask for this code. "
        "If you didn't request this, you can safely ignore this email.\n"
        "\n"
        "Best regards,\n"
        "The " + brandName + " Team\n"
    )

    html = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>" + escape(subject) + "</title></head>"
        '<body style="margin:0;padding:0;background:#f3f5f9;">'
        '<div style="display:none;max-height:0;overflow:hidden;opacity:0;">'
        "Your verification code is " + safeCode + ". It expires in " + str(minutes) + " minutes."
        "</div>"
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
        'style="background:#f3f5f9;padding:32px 12px;"><tr><td align="center">'
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
        'style="max-width:520px;background:#ffffff;border-radius:14px;overflow:hidden;'
        'font-family:Arial,Helvetica,sans-serif;color:#1f2937;">'
        '<tr><td style="background:#0f172a;padding:20px 32px;color:#ffffff;'
        'font-size:20px;font-weight:bold;letter-spacing:0.3px;">' + brandName + "</td></tr>"
        '<tr><td style="padding:32px 32px 8px 32px;">'
        '<h1 style="margin:0 0 16px 0;font-size:22px;color:#0f172a;">Verify your '
        + brandName + " account</h1>"
        '<p style="margin:0 0 12px 0;font-size:15px;line-height:1.6;">Hello there,</p>'
        '<p style="margin:0 0 20px 0;font-size:15px;line-height:1.6;">Please enter the '
        "following verification code to complete your sign-in:</p>"
        '<div style="margin:0 0 20px 0;padding:18px;text-align:center;background:#eef4ff;'
        "border:1px solid #cfe0ff;border-radius:10px;font-family:'Courier New',monospace;"
        'font-size:34px;font-weight:bold;letter-spacing:10px;color:#0f172a;">'
        + safeCode + "</div>"
        '<p style="margin:0 0 20px 0;font-size:15px;line-height:1.6;">This code will expire in '
        "<strong>" + str(minutes) + " minutes</strong>.</p>"
        '<p style="margin:0 0 24px 0;padding:12px 14px;font-size:13px;line-height:1.6;'
        'background:#f8fafc;border-left:3px solid #94a3b8;color:#475569;">'
        "<strong>Security tip:</strong> " + brandName + " will never call or message you to "
        "ask for this code. If you didn't request this, you can safely ignore this email.</p>"
        '<p style="margin:0 0 4px 0;font-size:15px;line-height:1.6;">Best regards,</p>'
        '<p style="margin:0 0 24px 0;font-size:15px;line-height:1.6;font-weight:bold;">The '
        + brandName + " Team</p>"
        "</td></tr>"
        '<tr><td style="padding:16px 32px 24px 32px;border-top:1px solid #e5e7eb;'
        'font-size:12px;line-height:1.5;color:#94a3b8;">This is an automated message. '
        "Please do not reply to this email.</td></tr>"
        "</table></td></tr></table></body></html>"
    )
    return subject, text, html