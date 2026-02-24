# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.6.x   | Yes                |
| < 0.6   | No                 |

## Reporting a Vulnerability

If you discover a security vulnerability in casual-llm, please report it responsibly.

**Do not** open a public GitHub issue for security vulnerabilities.

### How to Report

1. **GitHub Security Advisories** (preferred): Use [GitHub's private vulnerability reporting](https://github.com/casualgenius/casual-llm/security/advisories/new) to create a confidential advisory.

2. **Email**: Contact the maintainer directly at alex@casualgenius.com with:
   - A description of the vulnerability
   - Steps to reproduce
   - Potential impact assessment
   - Any suggested fixes (optional)

### What to Expect

- **Acknowledgment** within 48 hours of your report
- **Assessment** within 1 week
- **Fix timeline** depends on severity:
  - Critical: Patch release within 1 week
  - High: Patch release within 2 weeks
  - Medium/Low: Next scheduled release

### Scope

The following are in scope:

- SSRF vulnerabilities in image fetching
- Parameter injection via `options.extra`
- API key exposure in logs or error messages
- Dependency vulnerabilities in core packages
- Authentication bypass or privilege escalation

### Out of Scope

- Vulnerabilities in provider APIs themselves (OpenAI, Anthropic, Ollama)
- Issues requiring physical access to the machine
- Social engineering attacks

## Security Considerations

See [docs/security.md](docs/security.md) for security best practices when using casual-llm.
