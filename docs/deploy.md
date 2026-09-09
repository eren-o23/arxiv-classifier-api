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

Do **not** run `make model` on the VM for a plain deploy. The Dockerfile
downloads the artifact from the Hub at the pinned revision during the build, so a
copy in `models/` would be 265 MB of unused disk.

The one exception is `scripts/eval_quant.py`, which scores the model outside the
container and does need a local copy — see
[numbers.md](numbers.md#int8-quantization--measured-not-shipped). It also needs a
python env the VM does not otherwise have (`uv` plus torch, transformers and
datasets from the CPU index). Both are fine to leave in place; neither is on the
serving path.

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

## Benchmarking the deployed box (M6)

The same two scripts M3 ran on the laptop, pointed at this VM. They are stdlib
only in `--url` mode and import nothing from `src/`, so the VM needs no venv and
no torch — plain `python3` is enough.

### `hey` is not installable the documented way

`bench/load.sh` needs it, it is not in Ubuntu's repos, and **the S3 links in
rakyll/hey's README are dead** (`AccessDenied`) while the GitHub releases carry
no binary assets at all. Build it in a throwaway Go container instead — Docker is
already here, so no toolchain is left behind on the box:

```bash
docker run --rm -v $HOME/.local/bin:/out golang:1-alpine \
  sh -c "go install github.com/rakyll/hey@latest && cp /go/bin/hey /out/hey"
docker rmi golang:1-alpine
export PATH=$HOME/.local/bin:$PATH      # the `deploy` user has no sudo
```

### Measure the service, not the internet

Two paths, and they answer different questions:

```bash
# The api container's address on the compose bridge. Plain HTTP, no Caddy, no
# TLS — like-for-like with the laptop's loopback numbers.
API=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' \
  $(docker compose ps -q api))

python3 bench/latency.py --url http://$API:8000 -n 500
URL=http://$API:8000 bench/load.sh

# What a user actually gets, Caddy and TLS included.
python3 bench/latency.py --url https://<name>.duckdns.org -n 200
```

The `expose`-not-`ports` invariant is untouched by this: the bridge network is
reachable from the VM's own host namespace, never from the internet.

**`bench/latency.py` opens a fresh connection per request** (stdlib `urllib`, no
keep-alive), so the through-Caddy run pays a TCP + TLS handshake the laptop
numbers never did. Its `model only` column is the one that compares. `hey` keeps
connections alive, so the sweeps are unaffected.

### Lift the rate limit first, and put it back

30/min would trip within seconds of a concurrency-16 sweep, and the result would
be a graph of Caddy's rejection rate rather than the service's throughput:

```bash
printf 'RATE_LIMIT_EVENTS=100000\n' >> .env
docker compose up -d          # not `make deploy` — no git pull, no xcaddy rebuild
# ... measure ...
grep -v '^RATE_LIMIT_EVENTS=' .env > .env.tmp && mv .env.tmp .env
docker compose up -d
```

Only the through-Caddy runs need this; the `$API` path never touches the
limiter. `docs/numbers.md` says which figures were taken with it lifted.

### Cold start and peak memory

`bench/latency.py --url` prints `n/a` for both, by design — it did not spawn the
server, so it cannot time its start or read its RSS. Measure them directly:

```bash
docker compose stop api
s=$(date +%s.%N); docker compose start api
until curl -sf -m 5 -X POST http://$API:8000/predict \
  -H 'content-type: application/json' -d '{"title":"x","abstract":"y"}' \
  >/dev/null 2>&1; do sleep 0.1; done
echo "cold start $(echo "$(date +%s.%N) - $s" | bc)s"

docker compose exec api cat /sys/fs/cgroup/memory.peak
```

`memory.peak` is cgroup v2's high-water mark and it is the right tool for the
same reason `getrusage` beat `ps` at M3: `docker stats` reports *current* usage.
Read it after a run that includes the batch pass, and before any restart — a
restart resets it. It is a cgroup figure, not the process RSS `bench/latency.py`
reports, so the two are comparable in spirit rather than identical.

## What the first deploy actually measured

Verified against `https://arxiv-classifier.duckdns.org` on 2026-09-09, on a
Hetzner CX23 (2 vCPU Xeon Skylake / 3.7 GB / 38 GB, Ubuntu 24.04, x86_64):

| Check | Result |
|---|---|
| `make smoke` over TLS | all 12 green, certificate verified (no `-k`) |
| Rate limit | 30 through the window, then 429 with `retry-after: 40` |
| `/health` under the same burst | 200 x 12 |
| 1.2 MB body, chunked | 413 |
| 1.2 MB body, Content-Length | 413 |
| Cert across `down`/`up` | no ACME request; loaded from `/data/caddy` |
| `/predict` latency | 108-557 ms, vs 59 ms p50 on the laptop |

The chunked case is the one `api.py`'s `limit_body` cannot catch — it caps on
Content-Length only — and closing it was the hardening job M4 deferred to here.
Both caps and the limiter were also verified locally against this exact
Caddyfile before the VM existed, with `DOMAIN=http://localhost` so no
certificate was requested.

**The latency spread is not a benchmark.** These are single curls over the
public internet from a laptop, including TLS handshake and round trip. M6
measures properly with `bench/latency.py --url` and replaces the README's
laptop numbers.

## Notes

- **Restarting Caddy resets the rate limiter.** State is in-memory, so a
  redeploy clears every client's window. Fine at this scale; it is also why the
  local test above had to recreate the container rather than wait out a window.
- **`HUB_REPO`/`REVISION` live in `.env`, regenerated by `make deploy`.** They
  are `${VAR:?}` in `docker-compose.yml` so a build can never run against an
  empty revision, and compose interpolates the whole file on *every* command —
  so passing them inline to `up` alone left `docker compose ps|logs|down`
  failing. Writing them to `.env` fixes every compose command, including the
  ones you type by hand when something is wrong.
- **A Makefile change takes two `make deploy` runs to land.** `git pull` happens
  *inside* the recipe, so make has already parsed the old Makefile for the run
  that fetches the new one. Only the target itself is affected; the images and
  compose files are current after the first run.
- **`hey` has no working published binary.** The S3 URLs its README gives return
  `AccessDenied` and the GitHub releases have no assets, so it is built from
  source in a `golang:1-alpine` container. Written up above rather than left as a
  surprise for whoever next runs a sweep.
