# OWASP Vulnerability Assessment Platform

Flask-based Employee Management Portal used as a realistic internal application for controlled security education. Team project for the "Web Attacks & Data Privacy" course.

Every attack lives in its own module with a **vulnerable** implementation and a **secure** implementation served side by side, using the same underlying data, so the effect of each mitigation is directly comparable.

## Project Structure

```text
Fןinal Project/
+-- app/
|   +-- __init__.py          application factory, blueprint registration
|   +-- auth.py               register / login / logout / dashboard
|   +-- database.py           SQLite schema + seed data
|   +-- portal.py             employee directory, announcements, messages, profile, settings
|   +-- modules.py            Security Labs hub + SQL Injection module
|   +-- xss.py                Cross-Site Scripting module
|   +-- csrf.py                Cross-Site Request Forgery module
|   +-- ssrf.py                Server-Side Request Forgery module + /internal simulated service
|   +-- detection.py           statistics-based threat detection (request logging + flags)
+-- attacks/
|   +-- payloads.py            predefined attack payloads
|   +-- automation.py          attack automation script (run against the live server)
+-- config.py
+-- run.py
+-- requirements.txt
+-- static/
|   +-- css/style.css
|   +-- attacker/csrf_transfer_poc.html   standalone cross-site CSRF proof-of-concept page
+-- templates/
    +-- auth/, portal/, modules/ (sql_injection/, xss/, csrf/, ssrf/, threat_detection/)
```

## Windows Setup

Open PowerShell in the project folder:

```powershell
cd "path\to\Fןinal Project"
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation scripts:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run the Application

```powershell
python run.py
```

Open <http://127.0.0.1:5000>. The app creates `instance/owasp_platform.sqlite3` automatically on first run, seeded with sample employees, announcements, and a standing `attacker` account (password `attacker-lab-only`) used as the CSRF transfer target.

Register your own account, then open **Security Labs** in the sidebar (`/modules/`) to reach every module.

## Security Labs

| Lab | Vulnerable | Secure | What it shows |
| --- | --- | --- | --- |
| SQL Injection | `/modules/sql-injection/vulnerable-login` | `.../secure-login` | String-formatted vs. parameterized query; also `/portal/developer/employee-search` (unsafe) vs. `/portal/employees` (safe) |
| XSS | `/modules/xss/vulnerable-wall` | `/modules/xss/secure-wall` | Same stored comments, rendered unescaped vs. auto-escaped (+ CSP header). Cookie-theft demo collects at `/modules/xss/collect`, viewable at `/modules/xss/stolen` |
| CSRF | `/modules/csrf/vulnerable-transfer` | `/modules/csrf/secure-transfer` | A "send points" wallet feature with no CSRF token (and GET support) vs. a POST-only endpoint requiring a per-session synchronizer token |
| SSRF | `/modules/ssrf/vulnerable-fetch` | `/modules/ssrf/secure-fetch` | Unvalidated server-side URL fetch that can reach `/internal/secret-report` (a route that only accepts requests from localhost) vs. a host allow-list + private-IP-blocking fetch |
| Threat Detection | `/modules/threat-detection/` | - | Live dashboard over every logged request: total volume, top requesting IPs, failed-login clusters, and requests flagged for SQLi/XSS/path-traversal keyword patterns |
| Attack Automation | `/modules/attack-automation` | - | Results table populated by `attacks/automation.py` |

### Demonstrating the CSRF attack cross-site

`static/attacker/csrf_transfer_poc.html` is a standalone page that is **not** served by the Flask app - open it directly from disk (or serve it from a different port with `python -m http.server 8000`) while logged in to the portal in the same browser. It fires a forged transfer at the vulnerable endpoint using an `<img>` GET request, which works under the browser's default `SameSite=Lax` cookie policy with no user interaction. This is the piece of the demo that needs to happen in an actual browser, not the automation script (see below).

### Running the attack automation script

With the server running (`python run.py` in one terminal), in another terminal:

```powershell
cd attacks
..\.venv\Scripts\python.exe automation.py
```

It self-registers/logs in a dedicated `autobot` account and fires every payload in `payloads.py` at the SQLi, XSS, CSRF, and SSRF endpoints, printing each outcome and recording it in the `attack_results` table (view at `/modules/attack-automation`). It performs:

- **SQLi**: a dictionary of login-bypass strings, one UNION-based payload that exfiltrates `users.password_hash` through the vulnerable employee search, and a boolean-based **blind** extraction that recovers the `attacker` account's password hash prefix character-by-character with no visible query output.
- **XSS**: posts payloads to both walls and confirms the vulnerable one stores them unescaped (executable) while the secure one neutralizes them. It cannot trigger the browser-only cookie-theft `fetch()` call itself (plain HTTP requests don't execute JavaScript) - open the vulnerable wall in a real browser to see that part fire and land on the Attacker Loot page.
- **CSRF**: confirms the vulnerable endpoint performs the transfer with no token check. The actual cross-site delivery is what the PoC HTML page above demonstrates in a browser.
- **SSRF**: fetches the internal-only report URL and a public URL through the vulnerable fetcher.

### Threat detection

`app/detection.py` logs every request (method, path, IP, user agent) into `request_logs`, and flags requests whose path/form data match known SQLi/XSS/path-traversal patterns into `security_alerts`, plus every failed login attempt. This runs automatically for all traffic, including Burp Suite and the automation script, so `/modules/threat-detection/` fills in as you use the other labs.

## Notes

- Local educational use only; never deploy these vulnerable modules to the public internet.
- `config.py` stays `DEBUG=True` with a hardcoded `SECRET_KEY` intentionally - the project's own assumptions state it never leaves a controlled local environment.
- Local SQLite database files are ignored by Git.
