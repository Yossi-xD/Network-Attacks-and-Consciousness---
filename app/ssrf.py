import ipaddress
import socket
from urllib.parse import urlparse

import requests
from flask import Blueprint, render_template, request

from app.auth import login_required


ssrf_bp = Blueprint("ssrf", __name__, url_prefix="/modules/ssrf")

# Mitigation for secure_fetch(): only these hostnames may be fetched at all.
# An allow-list beats a block-list here because new internal hostnames can
# appear later, but this set only ever grows by deliberate choice.
ALLOWED_HOSTS = {"example.com", "httpbin.org", "jsonplaceholder.typicode.com"}


def _is_safe_url(url):
    """Reject anything but an allow-listed https(s) host that resolves to a
    public IP address. Checking the resolved IP (not just the hostname)
    stops DNS-rebinding-style tricks where an allowed name points at a
    private address."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False, "Could not parse URL."

    if parsed.scheme not in ("http", "https"):
        return False, "Only http/https URLs are allowed."

    hostname = parsed.hostname
    if not hostname:
        return False, "URL is missing a hostname."

    if hostname not in ALLOWED_HOSTS:
        return False, f"'{hostname}' is not on the allow-list."

    try:
        resolved_ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        return False, "Hostname could not be resolved."

    ip_obj = ipaddress.ip_address(resolved_ip)
    if (
        ip_obj.is_private
        or ip_obj.is_loopback
        or ip_obj.is_link_local
        or ip_obj.is_reserved
        or ip_obj.is_multicast
    ):
        return False, f"Resolved IP {resolved_ip} is not a public address."

    return True, None

# A route that simulates an internal-only admin endpoint - the kind of thing
# a real deployment would firewall off from the public internet but leave
# reachable from the app server itself (e.g. a cloud metadata endpoint, an
# internal admin API, a Kubernetes service). SSRF matters because it lets an
# outside attacker reach this indirectly through the app server.
internal_bp = Blueprint("internal", __name__, url_prefix="/internal")

LOCAL_ADDRESSES = {"127.0.0.1", "::1", "localhost"}


@internal_bp.route("/secret-report")
def secret_report():
    """Return sensitive data, but only to callers whose IP is localhost.

    A direct browser/curl request from anywhere but the server itself gets a
    403. The vulnerable SSRF fetcher below is a way around that check,
    because when the *Flask server* makes the request, it originates from
    127.0.0.1 - exactly like a real SSRF against an internal-only service.
    """
    if request.remote_addr not in LOCAL_ADDRESSES:
        return "Forbidden: this endpoint only accepts requests from localhost.", 403

    return (
        "CONFIDENTIAL INTERNAL REPORT\n"
        "-----------------------------\n"
        "Admin API key: LAB-DEMO-KEY-1234567890\n"
        "Internal notes: Q3 payroll adjustments pending finance sign-off.\n"
        "This page is normally unreachable from outside the server host.\n"
    )


@ssrf_bp.route("/")
@login_required
def index():
    """Show the SSRF lab overview."""
    return render_template("modules/ssrf/index.html")


@ssrf_bp.route("/vulnerable-fetch", methods=("GET", "POST"))
@login_required
def vulnerable_fetch():
    """Fetch any user-supplied URL from the server side with no validation.

    WARNING: This intentionally fetches whatever URL is submitted, including
    internal-only addresses, for local education only. A real app must never
    let user input choose the destination of a server-side request without
    validating it against an allow-list (see the secure-fetch mitigation).
    """
    target_url = None
    result = None

    if request.method == "POST":
        target_url = request.form.get("url", "").strip()
        if target_url:
            try:
                response = requests.get(target_url, timeout=3)
                result = {
                    "success": True,
                    "status_code": response.status_code,
                    "body": response.text[:2000],
                }
            except requests.RequestException as exc:
                result = {"success": False, "error": str(exc)}

    return render_template(
        "modules/ssrf/vulnerable_fetch.html", target_url=target_url, result=result
    )


@ssrf_bp.route("/secure-fetch", methods=("GET", "POST"))
@login_required
def secure_fetch():
    """Fetch a URL only if it passes the host allow-list and IP checks."""
    target_url = None
    result = None

    if request.method == "POST":
        target_url = request.form.get("url", "").strip()
        if target_url:
            is_safe, reason = _is_safe_url(target_url)
            if not is_safe:
                result = {"success": False, "error": f"Blocked: {reason}"}
            else:
                try:
                    # allow_redirects=False stops an allow-listed host from
                    # redirecting the request to a private/internal address.
                    response = requests.get(target_url, timeout=3, allow_redirects=False)
                    result = {
                        "success": True,
                        "status_code": response.status_code,
                        "body": response.text[:2000],
                    }
                except requests.RequestException as exc:
                    result = {"success": False, "error": str(exc)}

    return render_template(
        "modules/ssrf/secure_fetch.html", target_url=target_url, result=result
    )
