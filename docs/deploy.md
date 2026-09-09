# Deploy (M5)

The service runs on one small VM: `docker compose` supervises two containers,
Caddy terminates TLS and is the only thing exposed to the internet. SPEC.md §9.

| | |
|---|---|
| VM | Hetzner Cloud **CX23** (Shared → Cost-Optimized → x86), ~$8.50/mo + ~$0.72 IPv4 |
| OS | Ubuntu 24.04 |
| Hostname | a free **DuckDNS** subdomain, real Let's Encrypt certificate |
| Deploy | `make deploy` on the VM = `git pull && docker compose up -d --build` |

x86_64 is not optional: `requirements.lock` is `--generate-hashes`-pinned for
`x86_64-unknown-linux-gnu`, so the build on the VM is native and `PLATFORM` is
left empty. Arm64 instances — Hetzner's Ampere types, Oracle's free tier — would
mean regenerating the lock, which is verified M4 work.

## One-time setup

### 1. The server

Hetzner Cloud console → **Add Server**:

- **Location** — nearest to you; it is the latency floor M6 measures against.
- **Image** — Ubuntu 24.04.
- **Type** — **Shared Resources** → the **Cost-Optimized** card → architecture
  **x86 (Intel®/AMD)** → **CX23**. The *Regular Performance* card next to it is
  the default selection and costs roughly three times as much. Do **not** pick
  **Arm64 (Ampere®)**, which is the other radio button in the same card.
- **Networking** — IPv4 enabled; DuckDNS needs an A record to point at.
- **SSH keys** — add your public key here. Without it Hetzner emails a root
  password and key-based login is not set up.

### 2. The hostname

At duckdns.org (GitHub login): add a subdomain, set its IP to the server's IPv4.
No token or cron job is needed — the IP is static.

**Check DNS before starting Caddy.** A certificate request against a wrong A
record fails and consumes one of Let's Encrypt's 5-per-week slots for that name:

```bash
dig +short <name>.duckdns.org      # must print the server's IP
```

### 3. Provision

```bash
ssh root@<ip>
apt update && apt install -y docker.io docker-compose-v2 git ufw

# Hetzner images ship with no swap. `pip install torch` in the image build is
# the realistic OOM point on a small box, and the xcaddy Go build is a second.
fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab

ufw allow 22,80,443/tcp && ufw enable
```

Docker publishes through its own iptables chain and bypasses ufw. That is
harmless here precisely because the only published ports are Caddy's 80 and 443
— the `api` service uses `expose`, never `ports`, so nothing else is reachable
from outside regardless of what ufw thinks.

### 4. Check out and configure

```bash
git clone https://github.com/eren-o23/arxiv-classifier-api.git
cd arxiv-classifier-api
printf 'DOMAIN=%s\n' '<name>.duckdns.org' > .env
```

`.env` is gitignored, so the hostname is not committed and `Caddyfile` is
identical on every box. `RATE_LIMIT_EVENTS` may also go here; it defaults to 30.

Do **not** run `make model` on the VM. The Dockerfile downloads the artifact
from the Hub at the pinned revision during the build; a copy in `models/` would
be 265 MB of unused disk.

### 5. Deploy

```bash
make deploy
```

## Verifying a deploy

```bash
URL=https://<name>.duckdns.org make smoke
```

The same gate M4 ran against the local container, unchanged — that is why
`scripts/smoke.py` is stdlib-only with no project imports. It polls `/health`
through the window where Caddy answers 502 because the API container is still
loading the model, which `_body()` tolerates by design.

Then, the things smoke does not cover:

```bash
# The SPEC §11 criterion, literally. No -k: the certificate must be real.
curl -X POST https://<name>.duckdns.org/predict \
  -H 'content-type: application/json' \
  -d '{"title":"...","abstract":"..."}'

# Rate limit fires on /predict and not on /health.
for i in $(seq 40); do curl -s -o /dev/null -w '%{http_code} ' -X POST \
  https://<name>.duckdns.org/predict -H 'content-type: application/json' \
  -d '{"title":"","abstract":"x"}'; done; echo

# Certificates survive a restart — i.e. the caddy_data volume is mounted.
docker compose down && docker compose up -d
docker compose logs caddy | grep -i 'certificate\|obtain'
```

Both caps and the limiter were verified locally against this exact Caddyfile
before first deploy (custom Caddy build, `DOMAIN=http://localhost`): 5/5 through
then 429 with `RATE_LIMIT_EVENTS=5`, `/health` unaffected across 12 requests, and
413 for a 1.2 MB body sent **both** chunked and with a Content-Length. The
chunked case is the one `api.py`'s `limit_body` cannot catch — it caps on
Content-Length only — and closing it was the hardening job M4 deferred to here.

## Notes

- **Restarting Caddy resets the rate limiter.** State is in-memory, so a
  redeploy clears every client's window. Fine at this scale; it is also why the
  local test above had to recreate the container rather than wait out a window.
- **`docker compose restart` fails without the env vars set.** `HUB_REPO` and
  `REVISION` are `${VAR:?}` in `docker-compose.yml` on purpose, so compose
  refuses rather than building against an empty revision. Go through
  `make deploy`, which exports both from the Makefile — the single place the
  SHA lives.
- **M6's load sweep must raise `RATE_LIMIT_EVENTS` first.** `bench/load.sh`
  sweeps to concurrency 16 and would trip a 30/min limit within seconds,
  producing a graph of Caddy's rejection rate rather than the service's
  throughput. Raise it in `.env`, redeploy, measure, put it back — and say in
  `docs/numbers.md` that the numbers were taken with the limit lifted.
