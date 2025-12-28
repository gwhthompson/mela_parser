# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

1. **Do NOT open a public issue** for security vulnerabilities
2. Email the maintainers directly or use GitHub's private vulnerability reporting feature
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fixes (optional)

### What to Expect

- **Acknowledgment**: We will acknowledge receipt within 48 hours
- **Assessment**: We will assess the vulnerability and determine its severity
- **Fix Timeline**: Critical issues will be addressed as soon as possible
- **Disclosure**: We will coordinate disclosure timing with you

## Security Measures

This project implements several security practices:

### Dependency Security

- **Automated scanning**: [pip-audit](https://github.com/pypa/pip-audit) checks for known vulnerabilities
- **Dependabot**: Automatic dependency updates via GitHub Dependabot
- **Lock file**: Dependencies are pinned via `uv.lock`

### Code Security

- **Static analysis**: [Bandit](https://github.com/PyCQA/bandit) security linter in CI
- **Type checking**: Strict type checking reduces runtime errors
- **Input validation**: Pydantic models validate all external data

### CI/CD Security

- GitHub Actions workflows follow security best practices
- No secrets in code or logs
- OIDC authentication for PyPI publishing

## Best Practices for Users

When using mela-parser:

1. **Keep updated**: Use the latest version to get security fixes
2. **API keys**: Store OpenAI API keys securely (environment variables, not in code)
3. **Input validation**: Validate EPUB files before processing if from untrusted sources

## Security Updates

Security updates will be released as patch versions and announced via:

- GitHub Releases
- GitHub Security Advisories (for critical issues)
